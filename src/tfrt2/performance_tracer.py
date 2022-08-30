import threading
import os
import itertools
import queue
import pickle
import time
import traceback

import tensorflow as tf
import numpy as np
import PyQt5.QtCore as qtc

import tfrt2.tcp_base as tcp
import tfrt2.trace_engine as engine
import cumdistf.cdf as cdf


class JobReady(qtc.QObject):
    sig = qtc.pyqtSignal(tuple)


class PerformanceTracer:
    """
    This class performs high-performance ray tracing operations.

    This class attaches to a tracing_TCP_Server, and manages the various threads needed to run the
    time-consuming operations in an efficient and non-blocking manner.  It attempts to provide an interface to the
    tracing_TCP_server that is as clean and self-contained as possible.

    Jobs can be provided via queue_job(), which takes a string as the first argument that identifies which job to run,
    and any number of additional args that get passed to the function that actually accomplishes the job.  Results will
    be returned later, asynchronously.  Valid jobs are in the dictionary recognized_jobs.

    It is the responsibility of each job to communicate its results to the server.

    When a job is initiated, it sends a signal to server.busy_signal to notify the server that the optical system will
    be locked down, and should not be modified.  This lock-down is not enforced by the trace engine, it is the
    responsibility of the server to respect these signals.  New jobs can be sent while the server is busy - it uses
    a queue, but server synchronization cannot be performed while the engine is busy.  The engine also exposes a
    public attribute, busy, a multiprocessing.Event to indicate when it is running a job that locks the optical system.

    All jobs can be aborted with the function abort_jobs().  A ready signal is sent to server.busy_signal once the abort
    has fully completed.  Aborting may take time to complete, though it is nice if job functions provide a way to stop
    itself early when an abort is called for.  When a job aborts, it should typically not send any data to the server
    or client.  Jobs can poll for aborts via the multithreading.Event abort.

    While job_queue is a public attribute (though most users will want to interface with it via queue_job instead,
    even though all it does is check that the job type is valid), _worker_queue is private.  _worker_queue provides
    a set of multiprocessing workers that can help parallelize each job.  Not every operation should be paralleled,
    but some certainly should be.  Each job is responsible for partitioning work among the workers.

    Unfortunately one cannot write to the server's socket from the various threads managed by this engine, so we
    have to use a pyqt signal: the server's send_message signal.

    """
    def __init__(self, server, optical_system, device_count, profiler_log_path=None):
        self.optical_system = optical_system
        self.server = server
        self.profiler_log_path = profiler_log_path
        self.step_number = 0

        self.recognized_jobs = {
            "illuminance_plot": self._illuminance_plot,
            "full_ray_trace": self._full_ray_trace,
            "single_step": self._single_step,
        }

        # analyze the system and define the devices to use.  Each device will get its own process.  GPUs are
        # prioritized, if they exist.  Will create one process per device, unless the user specifies to use a different
        # number of devices.
        self.cpu_devices = tf.config.list_physical_devices("CPU")
        self.gpu_devices = tf.config.list_physical_devices("GPU")
        devices = self.gpu_devices or self.cpu_devices  # prioritize using gpu devices, if there are any
        if device_count > 0:
            # User requested a specific number of devices, so apportion them accordingly
            devices = [d for i, d in zip(range(device_count), itertools.cycle(devices))]
        self.devices = [d.name.replace("physical_", "").lower() for d in devices]
        self.cpu_device = self.cpu_devices[0].name.replace("physical_", "").lower()
        print(self.system_report())

        # An event that will be used to determine whether the engine is busy, which will block access to the optical
        # system and prevent synchronization requests.
        self.busy = threading.Event()
        self.BUSY_TIMEOUT = .25
        self.busy.clear()

        # A signal to end processing loops, to help gracefully shut down.  Though I admit I have no idea
        # if this actually does anything.
        self._shutdown = threading.Event()
        self._shutdown.clear()

        # A signal to abort processing operations.
        self.abort = threading.Event()
        self.abort.clear()

        # Create a job queue to feed job requests to the engine management loop.  Any thread can add items to this
        # queue, and it should not block.
        self.job_queue = queue.Queue()

        # Create two private queues, for communicating with the worker processes
        self.worker_count = len(self.devices)
        self._sub_job_queue = queue.Queue(2 * self.worker_count)
        self._result_queue = queue.Queue()

        # Create a pyqt signal that will be fired whenever a job is ready.  I am pretty sure the engine manager thread
        # should be able to fire this without issue - it is a thread, not a process.
        self.job_ready = JobReady()

        # Create a thread (not process) to manage and process job requests.  This thread will be long-running and
        # frequently block, which is why it needs to be separated from the event loop that powers the server.
        self._engine_manager_thread = threading.Thread(target=self._engine_management_loop, daemon=True)
        self._engine_manager_thread.start()

        # Create worker threads for each device
        self.workers = []
        self.worker_processes = []
        for device in self.devices:
            worker = Worker(
                device, self._sub_job_queue, self._result_queue, self.abort, self._shutdown
            )
            thread = threading.Thread(target=worker.run, daemon=True)
            self.workers.append(worker)
            self.worker_processes.append(thread)
            thread.start()

    def try_wait(self):
        """
        Check whether the system is busy, wait for a short amount of time (self.BUSY_TIMEOUT), and if the system
        does not become ready, return False to indicate the request has failed.
        """
        return self.busy.wait(self.BUSY_TIMEOUT)

    def system_report(self):
        return (
            f"System has {os.cpu_count()} CPUs via os.cpu_count.  Via TF it has {len(self.cpu_devices)} CPU "
            f"devices: {self.cpu_devices} and {len(self.gpu_devices)} GPU devices:  {self.gpu_devices}. "
            f"The performance engine will use {len(self.devices)} devices: {self.devices}"
        )

    def _engine_management_loop(self):
        while not self._shutdown.is_set():
            job, args = self.job_queue.get(True)

            if job == "shutdown":
                break
            else:
                # Lock down the optical system / server
                self.busy.set()
                self.server.busy_signal.sig.emit(True)

                # Perform the job.  This operation will typically block for a long time
                if self.profiler_log_path is None:
                    self.recognized_jobs[job](*args)
                else:
                    with tf.profiler.experimental.Profile(self.profiler_log_path):
                        self.recognized_jobs[job](*args)

                # Unlock the optical system / server
                self.busy.clear()
                self.purge_jobs()
                self.abort.clear()
                self.server.busy_signal.sig.emit(False)

    def abort_jobs(self):
        self.abort.set()
        while True:
            try:
                self.job_queue.get(False)
            except queue.Empty:
                break

    def purge_jobs(self):
        while True:
            try:
                self._sub_job_queue.get(False)
            except queue.Empty:
                break
        while True:
            try:
                self._result_queue.get(False)
            except queue.Empty:
                break

    def shut_down(self):
        self._shutdown.set()
        self.job_queue.put(("shutdown", None))

    def nonfatal_error(self, context, ex=None):
        if self.optical_system is None:
            context = context + "\nOptical system: Not loaded / synchronized."
        else:
            context = context + "\nOptical system: Loaded."

        if ex is None:
            ex = str(traceback.format_exc())
        else:
            ex = str(ex)

        self.server.send_data(tcp.SERVER_ERROR, pickle.dumps((context, ex)))
        print("Nonfatal exception raised: " + context)
        print(ex)
        print("--End traceback--")

    def queue_job(self, job_type, *args):
        if job_type in self.recognized_jobs.keys():
            if len(args) >= 1:
                self.job_queue.put((job_type, args))
            else:
                self.job_queue.put((job_type, tuple()))

    def send_status_update(self, message, current_step, total_steps):
        self.server.send_data(tcp.SERVER_ST_UPDATE, pickle.dumps((message, current_step, total_steps)))

    def _illuminance_plot(
        self, ray_count, ray_count_factor, x_res, y_res, x_min, x_max, y_min, y_max, standalone_plot
    ):
        """
        Trace many rays to generate an illuminance plot of the output.  This function is designed to be able to be
        called inline during optimization to update the goal flattener, but can also be used to generate illuminance
        plots for the client.

        This function will update always the goal flattening icdf, if appropriate, whether or not standalone_plot is
        True.

        Will only use finished rays to make the histogram, and does not care about the number of rays that actually
        finish.  If no rays finish, this function will happily return with a blank histogram and no warning.

        Parameters
        ----------
        ray_count : int
            The minumum number of rays to trace when forming this illuminance plot.  This is a minimum to allow the
            ray count used in each cycle to vary dynamically, and because I don't see the need to break the cycles up
            so that they add up to a precise value.  We are making a histogram.  More data is better.
        ray_count_factor : float
            A factor used to scale the rays generated by the sources, to increase batch size and reduce batch number.
            More performant if as high as possible without causing OOM errors.
        x_res, y_res : int, optional
            The resolution of the histogram to make.  Ignored if standalone_plot is False, required if True.
        x_min, x_max, y_min, y_max : int, optional
            The limits in which the histogram is evaluated.  Values outside these limits will be clipped to fit within
            them.
        standalone_plot : bool
            If True, will use the x and y parameters to define the histogram, and will send this histogram to the
            client.  If False, will ignore the x and y parameters and use the system goal to histogram the data.
        """
        try:
            # Update the optical system, and capture its boundary data.  This can be re-used for every iteration.
            results = self._sub_job_management(ray_count, ray_count_factor, "fast", "Measuring Illuminance.")
            if results is None:
                return

            processed_results = []
            for r in results:
                if type(r) is np.ndarray:
                    processed_results.append(r)

            full_results = np.concatenate(processed_results, axis=0)
            if standalone_plot:
                # Always make our own histogram, if standalone_plot was requested.  It could be a repeat of the
                # operations to build the flattener, down below, but I don't think it is worth the trouble to check
                # whether all the parameters between this plot and that plot are the same, which is what it would take
                # to use the results in both locations.
                histo, _, _ = np.histogram2d(
                    full_results[:, 0],
                    full_results[:, 1],
                    bins=(x_res, y_res),
                    range=((x_min, x_max), (y_min, y_max))
                )
                self.server.send_data(tcp.SERVER_ILUM, pickle.dumps(histo))

            # Always feed this data to the system goal, if appropriate, and send this data to the client too, as a
            # separate message.
            if self.optical_system.goal is not None:
                flat_result = self.optical_system.goal.feed_flatten(full_results)
                if flat_result is not None:
                    self.server.send_data(tcp.SERVER_FLATTENER, pickle.dumps(flat_result))

            # Clear the data that was prepped in the workers
            for worker in self.workers:
                worker.prepped_data = {}
        except Exception:
            self.nonfatal_error("Making illuminance plot")

    def _full_ray_trace(self, ray_count_factor):
        """
        Perform a full ray_trace and return all ray sets.

        This operation takes no arguments because it uses the settings.  It does not use workers because this operation
        is only intended to be used for testing, debugging, and display purposes.  It will frequently use only a small
        number of rays, and I suspect that the overhead of invoking the worker processes will frequently not be worth
        the extra power.
        """
        try:
            self.optical_system.update(ray_count_factor)
            self.optical_system.ray_trace(self.server.settings.trace_depth)
            self.server.send_data(tcp.SERVER_TRACE_RSLT, pickle.dumps(self.optical_system.get_raysets()))
        except Exception:
            self.nonfatal_error("ray trace for client")

    def _sub_job_management(
        self, ray_count, ray_count_factor, sub_job_key, status_message=None, add_extra=None
    ):
        """
        Manage distributing system and ray data to the sub_job queue and pulling back results.
        """
        with tf.device(self.cpu_device):
            self.optical_system.update(update_sources=False)
            boundary_data = self.optical_system.fuse_boundaries()
            if boundary_data[0].shape[0] == 0:
                self.nonfatal_error("There are no optical boundaries.")
                return

            # This data will not change run to run, so it can be sent to the workers now
            for i in range(len(self.workers)):
                self.workers[i].prep_data(
                    boundary_data=boundary_data,
                    trace_depth=tf.constant(self.server.settings.trace_depth, dtype=tf.int32),
                    intersect_epsilon=self.optical_system.intersect_epsilon,
                    size_epsilon=self.optical_system.size_epsilon,
                    ray_start_epsilon=self.optical_system.ray_start_epsilon,
                    new_ray_length=self.optical_system.new_ray_length,
                )

        init = True
        traced_rays = 0
        cached_source_data = None
        results = []
        pending_results = 0

        while traced_rays < ray_count and not self.abort.is_set():
            # Get a sample of rays from the optical system, but only if we haven't already generated a cache of rays
            # Are caching rays because we want to be able to place them into the queue without blocking, and want to
            # save them for later if we cannot put them
            if cached_source_data is None:
                with tf.device(self.cpu_device):
                    source_rays, extra = self.optical_system.get_ray_sample(ray_count_factor, add_extra)
                    source_ray_data = source_rays.prepare_for_tracer(self.optical_system.materials, extra)
                    if source_rays.count == 0:
                        self.nonfatal_error("There are no sources, or they are all disabled.")
                        return
            else:
                # pull the cached rays to use for this iteration
                source_ray_data = cached_source_data
                cached_source_data = None

            try:
                self._sub_job_queue.put((sub_job_key, source_ray_data), False)
                pending_results += 1
                traced_rays += source_rays.count
            except queue.Full:
                cached_source_data = source_ray_data
                init = False

            # Initially, (when init == True), lets continue here to fill up the sub_job_queue until it is totally full
            # once it fills, the above exception clause will fire, flipping the init flag, disabling this continue
            # for the rest of the loop.  Now, we will try to put a single new set every time one is removed, until
            # we don't need to put any more sets, at which points the loop will terminate.
            if init:
                continue

            # Get a result.  This call is blocking if no results are available - results coming available will set the
            # speed at which this loop runs.  Impossible to get here without any results available, because we always
            # put an element on the sub_job queue before getting here.  And since we started off filling the sub job
            # queue, we will always have an excess of results.  The loop will terminate before we run out of results
            # here.  But that means we need an extra loop to pull off the last results
            result = self._result_queue.get(True)
            pending_results -= 1
            # If the result is an error, need to send the error message and break out of the loop
            if isinstance(result, SubJobError):
                self.nonfatal_error("Illuminance plot sub-job", result)
                break
            if status_message is not None:
                self.send_status_update(status_message, min(traced_rays, ray_count), ray_count)
            results.append(result)

        # The loop has exited, meaning there are no more sub_jobs to enqueue.  But there are still results waiting,
        # so eat them until the results queue is empty.
        # The queue.get call inside this loop has to block, because we need to wait until results are empty, but we
        # also have to have some way of knowing when there are no more results and to move on to the final step.
        # My initial design was to make this call not block and look for queue empty, but this won't work if results
        # get dequeued faster than they are produced (which will be typical).  So instead I have to count
        # pending requests to make sure they all get processed.
        # I initially had an and not abort.set() here, but can't.  This is the only place that knows how many results to
        # expect.  Need to block here until each result has been pulled, else could abort while a worker is working,
        # which could cause it to put its result up after we have aborted (and purged the result queue), which can
        # cause a wrong kind of result to be delivered to an operation.
        sent_result_error = False
        while pending_results > 0:
            result = self._result_queue.get(True)
            pending_results -= 1
            # If the result is an error, need to send the error message and return
            if isinstance(result, SubJobError):
                if not sent_result_error:
                    self.nonfatal_error("Illuminance plot sub-job", result)
                    sent_result_error = True
            else:
                if status_message is not None:
                    self.send_status_update(status_message, min(traced_rays, ray_count), ray_count)
                results.append(result)

        # If the loops above stopped because we aborted, then just return now
        if self.abort.is_set():
            return

        return results

    def _single_step(self, ray_count, ray_count_factor, display_sample_count, optimize_params, routine=False):
        try:
            if routine:
                status_message = None
            else:
                status_message = "Single Optimization Step."
            if len(self.optical_system.parameters) == 0:
                self.nonfatal_error("single step", "Optical system has no parameters.")
                return
            if self.optical_system.goal is None:
                self.nonfatal_error("single step", "Optical system has no goal.")
                return
            # Perform the trace pre-compilation
            results = self._sub_job_management(
                ray_count, ray_count_factor, "precompile_trace", status_message=status_message,
                add_extra=self.optical_system.goal.add_extra
            )

            if self.abort.is_set() or results is None:
                return

            # Results is a list of tuples.  Separate it out into three lists
            compiled_trig_map = []
            compiled_tm_indices = []
            compiled_s = []
            compiled_hat = []
            compiled_n = []
            compiled_meta = []
            compiled_wv = []
            compiled_extra = []
            compiled_finished_out = []

            for result in results:
                try:
                    source_rays, trig_map, tm_indices, finished_out = result
                    if self.optical_system.goal.add_extra is None:
                        s, hat, n, meta, wv = source_rays
                    else:
                        s, hat, n, meta, wv, extra, = source_rays
                        compiled_extra.append(extra)
                    compiled_s.append(s)
                    compiled_hat.append(hat)
                    compiled_n.append(n)
                    compiled_meta.append(meta)
                    compiled_wv.append(wv)
                    compiled_trig_map.append(trig_map)
                    compiled_tm_indices.append(tm_indices)
                    compiled_finished_out.append(finished_out)
                except ValueError:
                    print(f"just got a bad result while compiling: {result}")

            # Fuse the lists into tensors
            s = tf.concat(compiled_s, axis=0)
            hat = tf.concat(compiled_hat, axis=0)
            n = tf.concat(compiled_n, axis=0)
            wv = np.concatenate(compiled_wv, axis=0)
            meta = tf.concat(compiled_meta, axis=0)
            output = tf.concat(compiled_finished_out, axis=0)
            if self.optical_system.goal.add_extra is None:
                extra = tf.zeros((s.shape[0], 0), dtype=tf.float64)
            else:
                extra = tf.concat(compiled_extra, axis=0)
            trig_map, tm_indices = ragged_concatenate(compiled_trig_map, compiled_tm_indices)

            if self.abort.is_set():
                return

            # Run the optimization, and retrieve a sample of finished rays for display/debugging
            do_smooth, learning_rate, momentum, grad_clip_low, grad_clip_high = optimize_params
            ray_sample, grad_stats = self.optical_system.optimize_step(
                s, hat, n, meta, extra, trig_map, tm_indices,
                output,
                tf.convert_to_tensor(self.server.settings.trace_depth, dtype=tf.int32),
                learning_rate,
                momentum,
                (grad_clip_high, grad_clip_low, do_smooth)
            )

            if self.abort.is_set():
                return

            # Compile a sample of rays to send
            finished_s, finished_hat, finished_meta = ray_sample
            sample_indices = tf.random.shuffle(tf.range(finished_s.shape[0]))[:display_sample_count]
            finished_rays = (
                tf.gather(finished_s, sample_indices, axis=0).numpy(),
                tf.gather(finished_hat, sample_indices, axis=0).numpy(),
                wv[tf.gather(finished_meta, sample_indices, axis=0).numpy()]
            )

            # Compile the new, updated parameters
            params = [p.numpy() for p in self.optical_system.parameters]

            self.server.send_data(tcp.SERVER_SINGLE_STEP, pickle.dumps((finished_rays, params, grad_stats)))
        except cdf.ComputeRequiredError:
            self.nonfatal_error(
                "single step",
                "Goal has not had illuminance calculated (available under remote operations)."
            )
        except Exception:
            self.nonfatal_error("single step")


class Worker:
    def __init__(self, device, job_queue, result_queue, abort, shutdown):
        self.device = device
        self.job_queue = job_queue
        self.abort = abort
        self.shutdown = shutdown
        self.result_queue = result_queue
        self.last_time = time.time()
        self.prepped_data = {}

        self.job_LUT = {
            "fast": self._fast,
            "precompile_trace": self._precompile_trace,
        }

    def prep_data(self, **kwargs):
        self.prepped_data = kwargs

    def run(self):
        try:
            while not self.shutdown.is_set():
                job, args = self.job_queue.get(True)
                # t = time.time()  # TODO remote timeing
                # print(f"worker is starting a job {t - self.last_time} s after ending last one")
                # Always need to put something in the result queue every time something is removed from the job_queue,
                # even in the case of an error, to make sure things don't get out of sync.
                try:
                    self.result_queue.put(self.job_LUT[job](*args), True)
                except Exception:
                    self.result_queue.put(SubJobError(traceback.format_exc()), True)
                self.last_time = time.time()
                # print(f"worker just finished its job in {self.last_time-t} s")
        except KeyboardInterrupt:
            # Basically just to hide the error messages when it gets keyboard interrupt.  But there has to be a better
            # way, that also catches other termination methods...
            return

    def _fast(self, *source_data):
        with tf.device(self.device):
            ray_s, ray_hat, ray_meta = engine.fast_trace_loop(
                *source_data[:4],
                *self.prepped_data["boundary_data"],
                self.prepped_data["trace_depth"],
                self.prepped_data["intersect_epsilon"],
                self.prepped_data["size_epsilon"],
                self.prepped_data["ray_start_epsilon"],
                self.prepped_data["new_ray_length"]
            )
            return (ray_s + ray_hat).numpy()

    def _precompile_trace(self, *source_data):
        with tf.device(self.device):
            trig_map, trig_map_indices, finished_output = engine.trace_sample_loop(
                *source_data[:4],
                *self.prepped_data["boundary_data"],
                self.prepped_data["trace_depth"],
                self.prepped_data["intersect_epsilon"],
                self.prepped_data["size_epsilon"],
                self.prepped_data["ray_start_epsilon"],
                self.prepped_data["new_ray_length"]
            )
            return source_data, trig_map, trig_map_indices, finished_output


class SubJobError:
    def __init__(self, tb):
        self._traceback = str(tb)

    def __str__(self):
        return "Sub Job Error: " + str(self._traceback)


def ragged_concatenate(flat_list, index_list):
    """
    Concatenate a list of ragged tensors (which are just 2-tuples, flattened and indices) into a single ragged tensor.

    Parameters
    ----------
    flat_list : list of 1-D tensor
        A list of the flattened data for the ragged tensors
    index_list : list of 1-D tensor
        A list of the index tensors for each ragged tensor.

    Returns
    -------
    flattened : 1-D tensor
        The concatenated flattened ragged tensor
    indices : 1-D tensor
        The indices that slice the ragged tensor

    """
    ragged_row_count = int(tf.reduce_max([each.shape[0] for each in index_list])) - 1
    ragged_rows = [list() for i in range(ragged_row_count)]

    for flat, i in zip(flat_list, index_list):
        last_row = int(i.shape[0])
        for row in range(last_row - 2):
            ragged_rows[row].append(flat[i[row]:i[row + 1]])
        ragged_rows[last_row-2].append(flat[i[last_row-2]:])

    joined_rows = []
    indices = [0]
    position = 0
    for row in range(len(ragged_rows)):
        new_row = tf.concat(ragged_rows[row], axis=0)
        new_row_size = int(new_row.shape[0])
        position += new_row_size
        joined_rows.append(new_row)
        indices.append(position)
    return tf.concat(joined_rows, axis=0), tf.convert_to_tensor(indices, dtype=tf.int32)


