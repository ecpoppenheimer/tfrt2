import multiprocessing
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


class JobReady(qtc.QObject):
    sig = qtc.pyqtSignal(tuple)


class PerformanceTracer:
    """
    This class performs high-performance ray tracing operations.

    This class attaches to a tracing_TCP_Server, and manages the various threads and processes needed to run the
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
    def __init__(self, server, optical_system, device_count):
        self.optical_system = optical_system
        self.server = server

        self.recognized_jobs = {
            "illuminance_plot": self._illuminance_plot,
            "full_ray_trace": self._full_ray_trace,
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
        self.busy = multiprocessing.Event()
        self.BUSY_TIMEOUT = .25
        self.busy.clear()

        # A signal to end processing loops, to help gracefully shut down.  Though I admit I have no idea
        # if this actually does anything.
        self._shutdown = multiprocessing.Event()
        self._shutdown.clear()

        # A signal to abort processing operations.
        self.abort = multiprocessing.Event()
        self.abort.clear()

        # Create a job queue to feed job requests to the engine management loop.  Any thread can add items to this
        # queue, and it should not block.
        self.job_queue = multiprocessing.Queue()

        # Create two private queues, for communicating with the worker processes
        self.worker_count = len(self.devices)
        self._sub_job_queue = multiprocessing.Queue(2 * self.worker_count)
        self._result_queue = multiprocessing.Queue()

        # Create a pyqt signal that will be fired whenever a job is ready.  I am pretty sure the engine manager thread
        # should be able to fire this without issue - it is a thread, not a process.
        self.job_ready = JobReady()

        # Create a thread (not process) to manage and process job requests.  This thread will be long-running and
        # frequently block, which is why it needs to be separated from the event loop that powers the server.
        self._engine_manager_thread = threading.Thread(target=self._engine_management_loop, daemon=True)
        self._engine_manager_thread.start()

        # Create worker processes for each device
        self.workers = []
        self.worker_processes = []
        for device in self.devices:
            worker = Worker(device, self._sub_job_queue, self._result_queue, self.abort, self._shutdown)
            process = multiprocessing.Process(target=worker.run, daemon=True)
            self.workers.append(worker)
            self.worker_processes.append(process)
            process.start()

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
                self.recognized_jobs[job](*args)

                # Unlock the optical system / server
                self.busy.clear()
                self.abort.clear()
                self.server.busy_signal.sig.emit(False)

    def abort_jobs(self):
        self.abort.set()
        while True:
            try:
                self.job_queue.get(False)
            except queue.Empty:
                break

    def shut_down(self):
        self._shutdown.set()
        self.job_queue.put(("shutdown", None))

    def nonfatal_error(self, context, ex=None):
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
            if len(args) > 0:
                self.job_queue.put((job_type, *args))
            else:
                self.job_queue.put((job_type, tuple()))

    def _illuminance_plot(
        self, ray_count, standalone_plot, x_res=None, y_res=None, x_min=None, x_max=None, y_min=None, y_max=None
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
        standalone_plot : bool
            If True, will use the x and y parameters to define the histogram, and will send this histogram to the
            client.  If False, will ignore the x and y parameters and use the system goal to histogram the data.
        x_res, y_res : int, optional
            The resolution of the histogram to make.  Ignored if standalone_plot is False, required if True.
        x_min, x_max, y_min, y_max : int, optional
            The limits in which the histogram is evaluated.  Values outside these limits will be clipped to fit within
            them.
        """
        # Update the optical system, and capture its boundary data.  This can be re-used for every iteration.
        with tf.device(self.cpu_device):
            self.optical_system.update()
            boundary_data = tuple(each.numpy() for each in self.optical_system.fuse_boundaries())

        init = True
        traced_rays = 0
        cached_source_rays = None
        results = []
        pending_results = 0

        while traced_rays < ray_count and not self.abort.is_set():
            # Get a sample of rays from the optical system, but only if we haven't already generated a cache of rays
            # Are caching rays because we want to be able to place them into the queue without blocking, and want to
            # save them for later if we cannot put them
            if cached_source_rays is None:
                with tf.device(self.cpu_device):
                    source_rays = self.optical_system.get_ray_sample(1)
                    source_rays = self.optical_system.evaluate_n(source_rays).numpy()
                    if source_rays.shape[0] == 0:
                        self.nonfatal_error("illuminance_plot, but there are no sources, or they are all disabled.")
                        return
            else:
                # pull the cached rays to use for this iteration
                source_rays = cached_source_rays
                cached_source_rays = None

            try:
                self._sub_job_queue.put(
                    (
                        "fast",
                        (
                            source_rays, *boundary_data, self.server.settings.trace_depth,
                            self.optical_system.intersect_epsilon.numpy(), self.optical_system.size_epsilon.numpy(),
                            self.optical_system.ray_start_epsilon.numpy(), self.optical_system.new_ray_length.numpy(),
                            self.optical_system.rayset_size
                         )
                    ),
                    False
                )
                pending_results += 1
                traced_rays += source_rays.shape[0]
            except queue.Full:
                cached_source_rays = source_rays
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
            # If the result is an error, need to send the error message and return
            if isinstance(result, SubJobError):
                self.nonfatal_error("Illuminance plot sub-job", result)
                return
            results.append(result)
            pending_results -= 1

        # The loop has exited, meaning there are no more sub_jobs to enqueue.  But there are still results waiting,
        # so eat them until the results queue is empty.
        # The queue.get call inside this loop has to block, because we need to wait until results are empty, but we
        # also have to have some way of knowing when there are no more results and to move on to the final step.
        # My initial design was to make this call not block and look for queue empty, but this won't work if results
        # get dequeued faster than they are produced (which will be typical).  So instead I have to count
        # pending requests to make sure they all get processed.
        while pending_results > 0 and not self.abort.is_set():
            result = self._result_queue.get(True)
            # If the result is an error, need to send the error message and return
            if isinstance(result, SubJobError):
                self.nonfatal_error("Illuminance plot sub-job", result)
                return
            results.append(result)
            pending_results -= 1

        # If the loops above stopped because we aborted, then just return now
        if self.abort.is_set():
            return

        # At this point we should have accumulated every result into the results list.  Just need to process and send
        # it on its way!
        full_results = np.concatenate(results, axis=0)
        if standalone_plot:
            # Always make our own histogram, if standalone_plot was requested.  Inefficient if
            histo, _, _ = np.histogram2d(
                full_results[:, 0],
                full_results[:, 1],
                bins=(x_res, y_res),
                range=((x_min, x_max), (y_min, y_max))
            )
            self.server.send_data(tcp.SERVER_ILUM, pickle.dumps(histo))

        # Always feed this data to the system goal, if appropriate, and send this data to the client too, as a separate
        # message.
        if self.optical_system.goal is not None:
            flat_result = self.optical_system.goal.feed_flatten(full_results)
            if flat_result is not None:
                self.server.send_data(tcp.SERVER_FLATTENER, pickle.dumps(flat_result))

    def _full_ray_trace(self):
        """
        Perform a full ray_trace and return all ray sets.

        This operation takes no arguments because it uses the settings.  It does not use workers because this operation
        is only intended to be used for testing, debugging, and display purposes.  It will frequently use only a small
        number of rays, and I suspect that the overhead of invoking the worker processes will frequently not be worth
        the extra power.
        """
        try:
            self.optical_system.update()
            self.optical_system.ray_trace(self.server.settings.trace_depth)
            all_raysets = self.optical_system.get_raysets(numpy=True)
            self.server.send_data(tcp.SERVER_TRACE_RSLT, pickle.dumps(all_raysets))
        except Exception:
            self.nonfatal_error("ray trace for client")


class Worker:
    def __init__(self, device, job_queue, result_queue, abort, shutdown):
        self.device = device
        self.job_queue = job_queue
        self.abort = abort
        self.shutdown = shutdown
        self.result_queue = result_queue

        self.job_LUT = {
            "fast": self._fast
        }

    def run(self):
        try:
            while not self.shutdown.is_set():
                job, args = self.job_queue.get(True)
                # Always need to put something in the result queue every time something is removed from the job_queue,
                # even in the case of an error, to make sure things don't get out of sync.
                try:
                    self.result_queue.put(self.job_LUT[job](*args), True)
                except Exception:
                    self.result_queue.put(SubJobError(traceback.format_exc()), True)
        except KeyboardInterrupt:
            # Basically just to hide the error messages when it gets keyboard interrupt.  But there has to be a better
            # way, that also catches other termination methods...
            return

    def _fast(
        self, source_rays, boundary_points, boundary_norms, metadata, trace_depth, intersect_epsilon, size_epsilon,
        ray_start_epsilon, new_ray_length, rayset_size
    ):
        with tf.device(self.device):
            source_rays = tf.convert_to_tensor(source_rays, dtype=tf.float64)
            boundary_points = tf.convert_to_tensor(boundary_points, dtype=tf.float64)
            boundary_norms = tf.convert_to_tensor(boundary_norms, dtype=tf.float64)
            metadata = tf.convert_to_tensor(metadata, dtype=tf.int32)
            trace_depth = tf.convert_to_tensor(trace_depth, dtype=tf.int32)
            intersect_epsilon = tf.convert_to_tensor(intersect_epsilon, dtype=tf.float64)
            size_epsilon = tf.convert_to_tensor(size_epsilon, dtype=tf.float64)
            ray_start_epsilon = tf.convert_to_tensor(ray_start_epsilon, dtype=tf.float64)
            new_ray_length = tf.convert_to_tensor(new_ray_length, dtype=tf.float64)

            output = engine.fast_trace_loop(
                source_rays, boundary_points, boundary_norms, metadata, trace_depth, intersect_epsilon, size_epsilon,
                ray_start_epsilon, new_ray_length, rayset_size
            )
            return output[:, 3:6].numpy()


class SubJobError:
    def __init__(self, tb):
        self._traceback = str(tb)

    def __str__(self):
        return "Sub Job Error: " + str(self._traceback)
