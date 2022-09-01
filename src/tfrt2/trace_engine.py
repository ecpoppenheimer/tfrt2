import tensorflow as tf
import numpy as np

from tfrt2.sources import RaySet3D

OPTICAL = 0
STOP = 1
TARGET = 2


class TraceEngine3D:
    all_raysets = {
        "active_rays", "finished_rays", "dead_rays", "stopped_rays", "all_rays", "unfinished_rays", "source_rays"
    }

    def __init__(self, materials):
        self.materials = materials
        self.active_rays = RaySet3D()
        self.finished_rays = RaySet3D()
        self.dead_rays = RaySet3D()
        self.stopped_rays = RaySet3D()
        self.all_rays = RaySet3D()
        self.unfinished_rays = RaySet3D()
        self.source_rays = RaySet3D()
        self.opticals = None
        self.stops = None
        self.targets = None

        self._new_ray_length = tf.constant(1.0, dtype=tf.float64)
        self._intersect_epsilon = tf.constant(1e-10, dtype=tf.float64)
        self._size_epsilon = tf.constant(1e-10, dtype=tf.float64)
        self._ray_start_epsilon = tf.constant(1e-10, dtype=tf.float64)

        self.optimizer = tf.keras.optimizers.SGD(1.0, 0.0, True)

    def clear_rays(self, clear_sources=True):
        self.active_rays = RaySet3D()
        self.finished_rays = RaySet3D()
        self.dead_rays = RaySet3D()
        self.stopped_rays = RaySet3D()
        self.all_rays = RaySet3D()
        self.unfinished_rays = RaySet3D()
        if clear_sources:
            self.source_rays = RaySet3D()

    def feed_raysets(self, all_sets):
        for key, value in all_sets.items():
            try:
                setattr(self, key, RaySet3D(*value))
            except KeyError:
                pass

    def get_raysets(self):
        return {key: RaySet3D.to_numpy(getattr(self, key)) for key in self.all_raysets}

    def fuse_boundaries(self):
        """
        The first part of ray tracing, this function collects data from the boundary parts and fuses them together into
        a single chunk in the correct format for the tracer.

        Rays also need to be fused, but this is taken care of in optical_system.refresh_source_rays, which is itself
        called by optical_system.update.  This function only does the boundaries

        All boundaries will get converted into three chunks:
            boundary_points : tf.float64 tensor of shape (n, 9), where n is the number of faces, and the second
                dimension contains the coordinates of the three vertices, in order first, second, third, where
                the first point is the 'pivot' or center vertex of the triangle, and second - first and third - first
                form the two basis vectors in which the intersection will be computed.
            boundary_norms : tf.float64 tensor of shape (n, 3), where n is the number of faces and the second dimension
                contains the norm vector of each face.
            metadata : tf.int32 tensor of shape (n, 3), where n is the number of faces and the three elements in
                the second dimension are material_in, material_out (indices to the system materials list) and
                the last element identifies which kind of boundary this is: optical = 0, target = 1, stop = 2.

        These three elements are returned in order.
        """

        boundaries = self.opticals + self.stops + self.targets

        if boundaries:
            p = tf.concat([b.p for b in boundaries], axis=0)
            u = tf.concat([b.u for b in boundaries], axis=0)
            v = tf.concat([b.v for b in boundaries], axis=0)
            n = tf.concat([b.norm for b in boundaries], axis=0)

            metadata = []
            for tp, box in zip(
                (OPTICAL, STOP, TARGET),
                (self.opticals, self.stops, self.targets)
            ):
                if box:
                    for b in box:
                        metadata.append(np.broadcast_to(((b.mat_in, b.mat_out, tp),), b.norm.shape))
            metadata = tf.cast(tf.concat(metadata, axis=0), tf.int32)
        else:
            p = tf.zeros((0, 3), dtype=tf.float64)
            u = tf.zeros((0, 3), dtype=tf.float64)
            v = tf.zeros((0, 3), dtype=tf.float64)
            n = tf.zeros((0, 3), dtype=tf.float64)
            metadata = tf.zeros((0, 3), dtype=tf.int32)

        return p, u, v, n, metadata

    def ray_trace(self, trace_depth):
        """
        Perform a full ray trace on the system, and fill the ray sets.

        This trace intersects every working ray with every boundary every iteration, so it is very resource-intensive,
        especially during backpropigation, but it also gives the most information about the system.  Ideally this
        function will only be used for display and debugging purposes.

        system.update() should typically be called before calling this, to re-randomize the source rays and update the
        optics, though this is not strictly necessary if you do not want to do this.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays
        """
        self.clear_rays(clear_sources=False)

        ray_data = self.source_rays.prepare_for_tracer(self.materials)
        boundary_data = self.fuse_boundaries()

        if ray_data[0].shape[0] == 0 or boundary_data[0].shape[0] == 0:
            # The system is empty.  Rays have already been cleared, so do nothing
            return

        active_s, active_hat, active_meta, finished_s, finished_hat, finished_meta, \
            stopped_s, stopped_hat, stopped_meta, dead_s, dead_hat, dead_meta, \
            unfinished_s, unfinished_hat, unfinished_meta = full_trace_loop(
                *ray_data[:4],
                *boundary_data,
                tf.convert_to_tensor(trace_depth, dtype=tf.int32),
                self._intersect_epsilon,
                self._size_epsilon,
                self._ray_start_epsilon,
                self._new_ray_length,
            )

        self.active_rays = RaySet3D(
            active_s, active_hat, tf.gather(self.source_rays.wv, active_meta), active_meta
        )
        self.finished_rays = RaySet3D(
            finished_s, finished_hat, tf.gather(self.source_rays.wv, finished_meta), finished_meta
        )
        self.stopped_rays = RaySet3D(
            stopped_s, stopped_hat, tf.gather(self.source_rays.wv, stopped_meta), stopped_meta
        )
        self.dead_rays = RaySet3D(
            dead_s, dead_hat, tf.gather(self.source_rays.wv, dead_meta), dead_meta
        )
        self.unfinished_rays = RaySet3D(
            unfinished_s, unfinished_hat, tf.gather(self.source_rays.wv, unfinished_meta), unfinished_meta
        )
        self.all_rays = RaySet3D.concatenate((
            self.active_rays, self.finished_rays, self.stopped_rays, self.dead_rays, self.unfinished_rays
        ))

    def fast_trace(self, trace_depth):
        """
        Perform a full ray trace on the system, but track only finished rays.

        This trace intersects every working ray with every boundary every iteration, so it is very resource-intensive,
        especially during backpropigation.  It will be marginally faster than ray_trace, because it eliminates
        everything except for compiling finished rays.  Does not even clear other ray sets.

        system.update() should typically be called before calling this, to re-randomize the source rays and update the
        optics, though this is not strictly necessary if you do not want to do this.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays
        """
        ray_data = self.source_rays.prepare_for_tracer(self.materials)
        boundary_data = self.fuse_boundaries()

        if ray_data[0].shape[0] == 0 or boundary_data[0].shape[0] == 0:
            # The system is empty.  Rays have already been cleared, so do nothing
            return

        finished_s, finished_hat, finished_meta = fast_trace_loop(
            *ray_data[:4],
            *boundary_data,
            tf.convert_to_tensor(trace_depth, dtype=tf.int32),
            self._intersect_epsilon,
            self._size_epsilon,
            self._ray_start_epsilon,
            self._new_ray_length,
        )
        self.finished_rays = RaySet3D(
            finished_s, finished_hat, tf.gather(self.source_rays.wv, finished_meta), finished_meta
        )

    def precompile_trace_samples(self, trace_depth):
        """
        Perform a full ray trace, but instead of compiling rays, compile the ray-boundary pairs for each intersection.

        The point of this function is to remove the very expensive process of determining which boundary each ray
        intersects with from the backpropigation step.  Full ray-boundary intersections require every ray-boundary pair
        to be computed, and backpropigation requires the full computation be stored in memory.  This makes full
        tracing under backpropigation extremely memory intensive, and limits ultimate achievable performance of the
        tracing engine.

        But there is absolutely no reason why the gradient needs to be computed for all the ray-boundary
        intersections that were not selected - this is just wasted computation.  This function allows these two steps
        to be separated: The full ray-boundary intersection is performed outside of the domain of gradient recording,
        and only the selected ray-boundary pairs are imported into the gradient, which greatly reduces the memory
        footprint required and therefore greatly enhances the capability of the optimizer.  The disadvantage is that
        the actual intersections have to be computed twice, once during this step and again during the forward
        phase of backpropigation.  But this is a small price to pay, because the second computation involves merely
        O(ray_count) operations rather than the O(ray_count * boundary_count) operations that a full trace requires.

        This function also makes for an excellent junction at which to parallelize computation: This function can be
        called in one thread / process / GPU / machine and the results computed here can be queued for another process
        to consume them.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays.

        Returns
        -------
        ray_data : tuple
            The ray data fed to the parameters.  Needed for the second half of the operation.
        trig_map , trig_map_indices : 1D int32 tensor
            The triangle (index into data in the boundary_data) that each ray (in ray_data) interacts with in each
            tracing iteration.  These two tensors are part of a pair; they should be a ragged tensor, but TF does not
            seem to support moving ragged tensors between eager and graph mode execution, so I have to do it myself.
            Fortunately it is easy:  trig_map is the actual data, flattened.  trig_map_indices is a flat tensor
            containing pairs of indices that define the slices into trig_map that form each row in the ragged tensor.
        """
        ray_data = self.source_rays.prepare_for_tracer(self.materials)
        boundary_data = self.fuse_boundaries()

        if ray_data[0].shape[0] == 0 or boundary_data[0].shape[0] == 0:
            return RaySet3D(), tf.zeros((0,), dtype=tf.int32), tf.zeros((2,), dtype=tf.int32)

        trig_map, trig_map_indices = trace_sample_loop(
            *ray_data[:4],
            *boundary_data,
            tf.convert_to_tensor(trace_depth, dtype=tf.int32),
            self._intersect_epsilon,
            self._size_epsilon,
            self._ray_start_epsilon,
            self._new_ray_length
        )
        return ray_data, trig_map, trig_map_indices

    def precompiled_trace(self, trace_depth, ray_data, boundary_data, trig_map, trig_map_indices):
        """
        Perform a trace from a precompiled sample, only tracking finished rays.

        This function is designed to be used with backpropigation; it requires a precompilation step
        (done with get_trace_samples() ) to determine which boundary each ray will intersect with at each step.
        That this function does not have to intersect every ray with every boundary means it is vastly more memory
        efficient.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays
        ray_data : tuple
            The ray data use to precompile the trace sample.  Obtained from RaySet3D.prepare_for_tracer.  But should
            feed the data returned from precompile_trace_samples.
        boundary_data : tuple
            Data like that obtained from optical_system.fuse_boundaries().  Must be identical to the data used by
            precompile_trace_samples, but... The variables that will be the target of optimization live inside the
            optical system, so optical_system.update() and optical_system.fuse_boundaries() must be called inside the
            gradient tape before calling this function.  These need to return the exact same data as was used to
            precompile the trace samples, so make sure to not update the parameters between when the trace samples are
            precompiled and calling this function.
        trig_map : 1D tf.int32 tensor
            The triangle that each ray intersects with.  See get_trace_samples for details.
        trig_map_indices : 1D tf.int32 tensor
            The slice indices into trig_map for each iteration.  See get_trace_samples for details.

        Returns
        -------
        A tuple of the s, hat, and meta fields of a rayset.  Does not return the wavelength, because that typically
        won't be needed, so cannot return as a RaySet3D.
        """
        if ray_data[0].shape[0] == 0:
            return (
               tf.zeros((0, 3), dtype=tf.float64), tf.zeros((0, 3), dtype=tf.float64), tf.zeros((0,), dtype=tf.int64)
            )

        finished_s, finished_hat, finished_meta = precompiled_trace_loop(
            *ray_data[:4],
            *boundary_data,
            trig_map,
            trig_map_indices,
            tf.convert_to_tensor(trace_depth, dtype=tf.int32),
            self._intersect_epsilon,
            self._new_ray_length
        )
        return finished_s, finished_hat, finished_meta

    def optimize_step(
        self, s, hat, n, meta, extra, trig_map, tm_indices, ray_output, trace_depth, learning_rate, momentum,
        process_params
    ):
        # Use the ray output compiled during the pre-compilation step to determine the goal.  They should be identical,
        # but using the precompiled data instead of the output generated by the following trace operation benefits
        # performance because the computations needed to generate the goal need to occur outside the gradient tape,
        # and will often occur outside tensorflow.  Separating them out allows use of tf.function around the gradient
        # operation.
        goal = tf.stop_gradient(tf.convert_to_tensor(self.goal.get_goal(ray_output, extra), dtype=tf.float64))

        grads, finished_s, finished_hat, finished_meta = self._compute_grad(
            s, hat, n, meta, trig_map, tm_indices, trace_depth, goal
        )
        grads_and_vars = []
        grad_stats = []
        for g, v, part in zip(grads, self.parameters, self.parametric_optics):
            processed_grad, mean, variance = self._process_grad(g, learning_rate, part, *process_params)
            grads_and_vars.append((processed_grad, v))
            grad_stats.append((part.name, mean, variance))

        self.optimizer.momentum = momentum
        self.optimizer.apply_gradients(grads_and_vars)

        return (finished_s, finished_hat, finished_meta), grad_stats

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    ])
    def _compute_grad(self, s, hat, n, meta, trig_map, tm_indices, trace_depth, goal):
        with tf.GradientTape() as tape:
            # Trace the system.  This is a tf.function decorated sub function, which MAY NOT WORK WITH THE TAPE!!!
            self.update_optics()
            finished_s, finished_hat, finished_meta = precompiled_trace_loop(
                s,
                hat,
                n,
                meta,
                *self.fuse_boundaries(),
                trig_map,
                tm_indices,
                trace_depth,
                self.intersect_epsilon,
                self.new_ray_length
            )

            traced_output = finished_s + finished_hat
            error = self.goal.error_function(traced_output, goal)
        return tape.gradient(error, self.parameters), finished_s, finished_hat, finished_meta

    @staticmethod
    def _process_grad(g, learning_rate, part, clip_high, clip_low, do_smooth):
        # Make sure that the gradient exists and is a number
        try:
            v = part.parameters
            if tf.reduce_any(tf.logical_not(tf.math.is_finite(g))):
                print(f"found nans! :(")
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        except ValueError:
            g = tf.zeros_like(v, dtype=tf.float64)
        g *= learning_rate * part.settings.relative_lr

        abs_g = tf.abs(g)
        pre_clip_mean = tf.reduce_mean(abs_g)
        pre_clip_variance = tf.math.reduce_variance(abs_g)

        # Clip the gradient relative to the value of each individual parameter.  Clip low is a magnitude.
        if clip_low > 0.0:
            clp_h = tf.abs(v * clip_high)
            clp_l = tf.abs(v * clip_low)
            is_positive = tf.greater(g, 0.0)
            grad_pos = tf.clip_by_value(g, clp_l, clp_h)
            grad_neg = tf.clip_by_value(g, -clp_l, -clp_h)
            g = tf.where(is_positive, grad_pos, grad_neg)
        else:
            clp_h = tf.abs(v * clip_high)
            g = tf.clip_by_value(g, -clp_h, clp_h)

        # Smooth the surface.  This is a global flag.  Smoothing must also be turned on for each component
        if do_smooth:
            part.smooth()

        # Apply a relative learning rate to this gradient.
        return g, pre_clip_mean, pre_clip_variance

    @property
    def new_ray_length(self):
        return self._new_ray_length

    @new_ray_length.setter
    def new_ray_length(self, val):
        self._new_ray_length = tf.constant(val, dtype=tf.float64)

    @property
    def intersect_epsilon(self):
        return self._intersect_epsilon

    @intersect_epsilon.setter
    def intersect_epsilon(self, val):
        self._intersect_epsilon = tf.constant(val, dtype=tf.float64)

    @property
    def size_epsilon(self):
        return self._size_epsilon

    @size_epsilon.setter
    def size_epsilon(self, val):
        self._size_epsilon = tf.constant(val, dtype=tf.float64)

    @property
    def ray_start_epsilon(self):
        return self._ray_start_epsilon

    @ray_start_epsilon.setter
    def ray_start_epsilon(self, val):
        self._ray_start_epsilon = tf.constant(val, dtype=tf.float64)


# ======================================================================================================================


@tf.function(input_signature=(
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.bool),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
))
def _select_intersections(ray_r, trig_u, trig_v, valid, size_epsilon, ray_start_epsilon):
    # prune invalid segment intersections
    valid = tf.logical_and(valid, tf.greater_equal(trig_u, -size_epsilon))
    valid = tf.logical_and(valid, tf.greater_equal(trig_v, -size_epsilon))
    valid = tf.logical_and(valid, tf.less_equal(trig_u + trig_v, 1 + size_epsilon))

    # prune invalid ray intersections
    valid = tf.logical_and(valid, tf.greater_equal(ray_r, ray_start_epsilon))

    # fill ray_r with large values wherever the intersection is invalid
    inf = 2 * tf.reduce_max(ray_r) * tf.ones_like(ray_r)
    ray_r = tf.where(valid, ray_r, inf)

    # find the closest ray intersection
    closest_trig = tf.argmin(ray_r, axis=1, output_type=tf.int32)
    valid = tf.reduce_any(valid, axis=1)
    ray_r = tf.gather(ray_r, closest_trig, axis=1, batch_dims=1)
    return valid, closest_trig, ray_r


@tf.function(input_signature=(
    RaySet3D.s_spec,
    RaySet3D.hat_spec,
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
))
def full_line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon):
    # Expand / meshgrid the ray and boundary data.  First dimension indexes rays, second dimension indexes boundaries.
    ray_count = tf.shape(ray_start)[0]
    boundary_count = tf.shape(boundary_p)[0]
    ray_start = tf.expand_dims(ray_start, 1)
    ray_start = tf.broadcast_to(ray_start, (ray_count, boundary_count, 3))
    ray_hat = tf.expand_dims(ray_hat, 1)
    ray_hat = tf.broadcast_to(ray_hat, (ray_count, boundary_count, 3))
    boundary_p = tf.expand_dims(boundary_p, 0)
    boundary_p = tf.broadcast_to(boundary_p, (ray_count, boundary_count, 3))
    boundary_u = tf.expand_dims(boundary_u, 0)
    boundary_u = tf.broadcast_to(boundary_u, (ray_count, boundary_count, 3))
    boundary_v = tf.expand_dims(boundary_v, 0)
    boundary_v = tf.broadcast_to(boundary_v, (ray_count, boundary_count, 3))

    return raw_line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon)


@tf.function(input_signature=(
    RaySet3D.s_spec,
    RaySet3D.hat_spec,
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
))
def line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon):
    """
    This stub of a function exists solely so I can ensure that raw_line_triangle_intersection is always wrapped
    in a tf.function with an input signature.  Difficult because it gets rank 3 tensors from
    full_line_triangle_intersection and rank 2 tensors from here.
    """
    return raw_line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon)


def dot(a, b):
    return tf.reduce_sum(a * b, axis=-1)


def raw_line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon):
    """
    Low-level function that actually computes the intersection.

    Math taken from https://www.sciencedirect.com/topics/computer-science/intersection-routine.  Finally, a clean
    vectorized implementation!

    The shape of every parameter must exactly match, except for epsilon, which is a scalar.

    Parameters
    ----------
    ray_start : (..., 3) float tensor
        The starting point of each ray
    ray_hat : (..., 3) float tensor
        The direction of each ray.  Does not need to be normalized.
    boundary_p : (..., 3) float tensor
        The pivot vertex on each triangle - the vertex around which the norm was computed.
    boundary_u, boundary_v : (..., 3) float tensor
        Two vectors defining the triangle - a vector from the other two vertices to the pivot vertex.
    epsilon : float
        A small value (I like 1e-10) used for numerical stability to avoid divide by zero.  This is used to catch
        line-plane pairs that are parallel or very nearly parallel, so if this is too large it can cause the algorithm
        to miss valid line-plane intersections that are very far away because the line and plane are very nearly
        parallel.
    """

    s_p = ray_start - boundary_p

    denominator = dot(tf.linalg.cross(ray_hat, boundary_v), boundary_u)
    ray_u_numerator = dot(tf.linalg.cross(s_p, boundary_u), boundary_v)
    trig_u_numerator = dot(tf.linalg.cross(ray_hat, boundary_v), s_p)
    trig_v_numerator = dot(tf.linalg.cross(s_p, boundary_u), ray_hat)

    # safe division
    valid = tf.greater_equal(tf.abs(denominator), epsilon)
    safe_value = tf.ones_like(denominator)
    safe_denominator = tf.where(valid, denominator, safe_value)
    ray_r = ray_u_numerator / safe_denominator
    trig_u = trig_u_numerator / safe_denominator
    trig_v = trig_v_numerator / safe_denominator

    return ray_r, trig_u, trig_v, valid


def old_line_triangle_intersection(ray_start, ray_hat, boundary_p, boundary_u, boundary_v, epsilon):
    """
    Low-level function that actually computes the intersection.  Algebra was done on wolfram alpha, so unfortunately
    not particularly readable, but it is just a solution to the parametric equation defined by
    ray = ray_start + r * (ray_end - ray_start)
    boundary = boundary_1 + u * (boundary_2 - boundary_1) + v * (boundary_3 - boundary_1)

    But I am copying the math over from tfrt1, which had the naming convention boundary_1 = p, boundary_2 = 1,
    boundary_3 = 2
    """
    rx1, ry1, rz1 = tf.unstack(ray_start, axis=-1)
    rx2, ry2, rz2 = tf.unstack(ray_hat + ray_start, axis=-1)
    xp, yp, zp, = tf.unstack(boundary_p, axis=-1)
    x1, y1, z1 = tf.unstack(boundary_p + boundary_u, axis=-1)
    x2, y2, z2 = tf.unstack(boundary_p + boundary_v, axis=-1)

    a = rx1 - rx2
    b = x1 - xp
    c = x2 - xp
    d = ry1 - ry2
    f = y1 - yp
    g = y2 - yp
    h = rz1 - rz2
    k = z1 - zp
    m = z2 - zp

    q = rx1 - xp
    r = ry1 - yp
    s = rz1 - zp

    denominator = a * g * k + b * d * m + c * f * h - a * f * m - b * g * h - c * d * k
    ray_u_numerator = b * m * r + c * f * s + g * k * q - b * g * s - c * k * r - f * m * q
    trig_u_numerator = a * g * s + c * h * r + d * m * q - a * m * r - c * d * s - g * h * q
    trig_v_numerator = a * k * r + b * d * s + f * h * q - a * f * s - b * h * r - d * k * q

    # safe division
    valid = tf.greater_equal(tf.abs(denominator), epsilon)
    safe_value = tf.ones_like(rx1)
    safe_denominator = tf.where(valid, denominator, safe_value)
    ray_r = ray_u_numerator / safe_denominator
    trig_u = trig_u_numerator / safe_denominator
    trig_v = trig_v_numerator / safe_denominator

    return ray_r, trig_u, trig_v, valid


@tf.function(input_signature=(
    RaySet3D.hat_spec,
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
))
def snells_law_3d(ray_hat, boundary_norms, n_in, n_out, new_ray_length):
    # A vector representing the ray direction.
    u = tf.math.l2_normalize(ray_hat, axis=1)

    # Need to normalize the norm (it isn't guaranteed to already be normed).
    n = tf.math.l2_normalize(boundary_norms, axis=1)
    nu = tf.reduce_sum(n * u, axis=1, keepdims=True)

    # process the index of refraction
    internal_mask = tf.greater(nu, 0.0)
    one = tf.ones_like(n_in)
    zero = tf.zeros_like(n_in)

    n_in_is_safe = tf.not_equal(n_in, 0.0)
    n_in_safe = tf.where(n_in_is_safe, n_in, one)
    n_out_is_safe = tf.not_equal(n_out, 0.0)
    n_out_safe = tf.where(n_out_is_safe, n_out, one)

    n1 = tf.reshape(tf.where(n_out_is_safe, n_in_safe / n_out_safe, zero), (-1, 1))
    n2 = tf.reshape(tf.where(n_in_is_safe, n_out_safe / n_in_safe, zero), (-1, 1))
    eta = tf.where(internal_mask, n1, n2)
    nu_eta = eta * nu

    # compute the refracted vector
    radicand = 1 - eta * eta + nu_eta * nu_eta
    do_tir = tf.less(radicand, 0)
    safe_radicand = tf.where(do_tir, tf.ones_like(radicand), radicand)
    refract = (tf.sign(nu) * tf.sqrt(safe_radicand) - nu_eta) * n + eta * u

    # compute the reflected vector
    reflect = -2 * nu * n + u

    # choose refraction or reflection
    reflective_surface = tf.reshape(tf.equal(n_in, 0), (-1, 1))
    do_reflect = tf.logical_or(do_tir, reflective_surface)
    new_vector = tf.where(do_reflect, reflect, refract)

    return new_ray_length * new_vector


# ======================================================================================================================
# Full trace, compile full ray sets, used primarily for display.


@tf.function
def full_trace_loop(
    ray_start, ray_hat, ray_n, ray_meta, boundary_p, boundary_u, boundary_v, boundary_norms, metadata, trace_depth,
    intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    counter = tf.constant(0, dtype=tf.int32)
    active_start = tf.zeros((0, 3), dtype=tf.float64)
    active_hat = tf.zeros((0, 3), dtype=tf.float64)
    active_meta = tf.zeros((0,), dtype=tf.int32)
    finished_start = tf.zeros((0, 3), dtype=tf.float64)
    finished_hat = tf.zeros((0, 3), dtype=tf.float64)
    finished_meta = tf.zeros((0,), dtype=tf.int32)
    dead_start = tf.zeros((0, 3), dtype=tf.float64)
    dead_hat = tf.zeros((0, 3), dtype=tf.float64)
    dead_meta = tf.zeros((0,), dtype=tf.int32)
    stopped_start = tf.zeros((0, 3), dtype=tf.float64)
    stopped_hat = tf.zeros((0, 3), dtype=tf.float64)
    stopped_meta = tf.zeros((0,), dtype=tf.int32)
    (
        active_start, active_hat, active_meta,
        finished_start, finished_hat, finished_meta,
        dead_start, dead_hat, dead_meta,
        stopped_start, stopped_hat, stopped_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    ) = tf.while_loop(
        _full_while_cond,
        _full_while_body,
        (
            active_start, active_hat, active_meta,
            finished_start, finished_hat, finished_meta,
            dead_start, dead_hat, dead_meta,
            stopped_start, stopped_hat, stopped_meta,
            ray_start, ray_hat, ray_n, ray_meta,
            boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
            counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
        ),
        shape_invariants=(
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, None)),
            tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(())
        )
    )

    return (
        active_start, active_hat, active_meta,
        finished_start, finished_hat, finished_meta,
        dead_start, dead_hat, dead_meta,
        stopped_start, stopped_hat, stopped_meta,
        working_start, working_hat, working_meta
    )


def _full_while_cond(
    active_start, active_hat, active_meta,
    finished_start, finished_hat, finished_meta,
    dead_start, dead_hat, dead_meta,
    stopped_start, stopped_hat, stopped_meta,
    working_start, working_hat, working_n, working_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
    counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_start)[0], 0))


def _full_while_body(
        active_start, active_hat, active_meta,
        finished_start, finished_hat, finished_meta,
        dead_start, dead_hat, dead_meta,
        stopped_start, stopped_hat, stopped_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    counter += 1
    # Outputs generated here are rectangular, because we are computing intersections between every ray
    # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
    ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
        working_start, working_hat, boundary_p, boundary_u, boundary_v, intersect_epsilon
    )
    valid, closest_trig, ray_r = _select_intersections(
        ray_r, trig_u, trig_v, valid, size_epsilon, ray_start_epsilon
    )

    # At this point, active, finished, dead, and stopped rays contain no rays from this iteration;
    # working rays contains every ray traced this iteration, and valid, closest_trig, and the parametric solution
    # all match working rays;
    # and boundary_points, boundary_norms, and metadata are all organized per triangle.

    # Any ray that does not have a valid intersection is a dead ray, so filter them out
    dead = tf.logical_not(valid)
    dead_start = tf.concat((dead_start, tf.boolean_mask(working_start, dead)), axis=0)
    dead_hat = tf.concat((dead_hat, tf.boolean_mask(working_hat, dead)), axis=0)
    dead_meta = tf.concat((dead_meta, tf.boolean_mask(working_meta, dead)), axis=0)

    # Filter out dead rays and project all remaining rays into their intersection.
    closest_trig = tf.boolean_mask(closest_trig, valid)
    ray_r = tf.reshape(tf.boolean_mask(ray_r, valid), (-1, 1))

    working_start = tf.boolean_mask(working_start, valid)
    working_hat = ray_r * tf.boolean_mask(working_hat, valid)
    working_n = tf.boolean_mask(working_n, valid)
    working_meta = tf.boolean_mask(working_meta, valid)

    # Select the interaction type by gathering out of metadata.
    intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)

    # Sort the rays into their bucket
    select_stopped = tf.equal(intersection_type, STOP)
    stopped_start = tf.concat((stopped_start, tf.boolean_mask(working_start, select_stopped)), axis=0)
    stopped_hat = tf.concat((stopped_hat, tf.boolean_mask(working_hat, select_stopped)), axis=0)
    stopped_meta = tf.concat((stopped_meta, tf.boolean_mask(working_meta, select_stopped)), axis=0)

    select_finished = tf.equal(intersection_type, TARGET)
    finished_start = tf.concat((finished_start, tf.boolean_mask(working_start, select_finished)), axis=0)
    finished_hat = tf.concat((finished_hat, tf.boolean_mask(working_hat, select_finished)), axis=0)
    finished_meta = tf.concat((finished_meta, tf.boolean_mask(working_meta, select_finished)), axis=0)

    select_working = tf.equal(intersection_type, OPTICAL)
    working_start = tf.boolean_mask(working_start, select_working)
    working_hat = tf.boolean_mask(working_hat, select_working)
    working_n = tf.boolean_mask(working_n, select_working)
    working_meta = tf.boolean_mask(working_meta, select_working)

    active_start = tf.concat((active_start, working_start), axis=0)
    active_hat = tf.concat((active_hat, working_hat), axis=0)
    active_meta = tf.concat((active_meta, working_meta), axis=0)

    # Gather the triangle data to match the working rays.
    closest_trig = tf.boolean_mask(closest_trig, select_working)
    selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
    mat_in = tf.gather(metadata_const[:, 0], closest_trig, axis=0)
    mat_out = tf.gather(metadata_const[:, 1], closest_trig, axis=0)

    # Determine the refractive index for each ray reaction
    n_in = tf.gather(working_n, mat_in, axis=1, batch_dims=1)
    n_out = tf.gather(working_n, mat_out, axis=1, batch_dims=1)

    # Perform the ray reactions.
    working_end = working_start + working_hat
    new_hat = snells_law_3d(
        working_hat, selected_boundary_norms, n_in, n_out, new_ray_length
    )

    return (
        active_start, active_hat, active_meta,
        finished_start, finished_hat, finished_meta,
        dead_start, dead_hat, dead_meta,
        stopped_start, stopped_hat, stopped_meta,
        working_end, new_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    )


# ======================================================================================================================
# Fast trace, only compiling finished rays, used primarily for analysing output.


@tf.function
def fast_trace_loop(
    ray_start, ray_hat, ray_n, ray_meta, boundary_p, boundary_u, boundary_v, boundary_norms, metadata, trace_depth,
    intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    counter = tf.constant(0, dtype=tf.int32)
    finished_start = tf.zeros((0, 3), dtype=tf.float64)
    finished_hat = tf.zeros((0, 3), dtype=tf.float64)
    finished_meta = tf.zeros((0,), dtype=tf.int32)
    (
        finished_start, finished_hat, finished_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    ) = tf.while_loop(
        _fast_while_cond,
        _fast_while_body,
        (
            finished_start, finished_hat, finished_meta,
            ray_start, ray_hat, ray_n, ray_meta,
            boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
            counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
        ),
        shape_invariants=(
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, None)),
            tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None, 3)),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(())
        )
    )

    return finished_start, finished_hat, finished_meta


def _fast_while_cond(
    finished_start, finished_hat, finished_meta,
    working_start, working_hat, working_n, working_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
    counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_start)[0], 0))


def _fast_while_body(
    finished_start, finished_hat, finished_meta,
    working_start, working_hat, working_n, working_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
    counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    counter += 1
    # Outputs generated here are rectangular, because we are computing intersections between every ray
    # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
    ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
        working_start, working_hat, boundary_p, boundary_u, boundary_v, intersect_epsilon
    )
    valid, closest_trig, ray_r = _select_intersections(
        ray_r, trig_u, trig_v, valid, size_epsilon, ray_start_epsilon
    )

    # Project all rays
    working_hat = tf.reshape(ray_r, (-1, 1)) * working_hat

    # Select and compile the finished rays
    intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)
    select_finished = tf.logical_and(valid, tf.equal(intersection_type, TARGET))

    finished_start = tf.concat((finished_start, tf.boolean_mask(working_start, select_finished)), axis=0)
    finished_hat = tf.concat((finished_hat, tf.boolean_mask(working_hat, select_finished)), axis=0)
    finished_meta = tf.concat((finished_meta, tf.boolean_mask(working_meta, select_finished)), axis=0)

    # Select the working rays
    select_working = tf.logical_and(valid, tf.equal(intersection_type, OPTICAL))
    working_start = tf.boolean_mask(working_start, select_working)
    working_hat = tf.boolean_mask(working_hat, select_working)
    working_n= tf.boolean_mask(working_n, select_working)

    # Gather the triangle data to match the working rays.
    closest_trig = tf.boolean_mask(closest_trig, select_working)
    selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
    mat_in = tf.gather(metadata_const[:, 0], closest_trig, axis=0)
    mat_out = tf.gather(metadata_const[:, 1], closest_trig, axis=0)

    # Determine the refractive index for each ray reaction
    n_in = tf.gather(working_n, mat_in, axis=1, batch_dims=1)
    n_out = tf.gather(working_n, mat_out, axis=1, batch_dims=1)

    # Perform the ray reactions.
    working_end = working_start + working_hat
    new_hat = snells_law_3d(
        working_hat, selected_boundary_norms, n_in, n_out, new_ray_length
    )

    return (
        finished_start, finished_hat, finished_meta,
        working_end, new_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    )


# ======================================================================================================================
# Precompiled trace, splitting ray-triangle intersection validation and tracing, primarily used in gradient descent.


@tf.function
def trace_sample_loop(
    ray_start, ray_hat, ray_n, ray_meta, boundary_p, boundary_u, boundary_v, boundary_norms, metadata, trace_depth,
    intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    matched_triangles = tf.zeros((0,), dtype=tf.int32)
    mt_counts = tf.zeros((0,), dtype=tf.int32)
    counter = tf.constant(0, dtype=tf.int32)
    finished_start = tf.zeros((0, 3), dtype=tf.float64)
    finished_hat = tf.zeros((0, 3), dtype=tf.float64)
    finished_meta = tf.zeros((0,), dtype=tf.int32)

    (
        finished_start, finished_hat, finished_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
        matched_triangles, mt_counts,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    ) = tf.while_loop(
        trace_sample_while_cond,
        trace_sample_while_body,
        (
            finished_start, finished_hat, finished_meta,
            ray_start, ray_hat, ray_n, ray_meta,
            boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
            matched_triangles, mt_counts,
            counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
        ),
        shape_invariants=(
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, None)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, None)),
            tf.TensorShape((None, None)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None,)), tf.TensorShape((None,)),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(())
        )
    )

    # These changes will turn counts into a more convenient form to slice out the needed data later:
    # to get layer n, slice from mt_counts[n] to mt_counts[n+1]
    mt_counts = tf.pad(mt_counts, ((1, 0),))
    mt_counts = tf.cumsum(mt_counts)
    return matched_triangles, mt_counts, finished_start + finished_hat


def trace_sample_while_cond(
    finished_start, finished_hat, finished_meta,
    working_start, working_hat, working_n, working_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
    matched_triangles, mt_counts,
    counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_start)[0], 0))


def trace_sample_while_body(
    finished_start, finished_hat, finished_meta,
    working_start, working_hat, working_n, working_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
    matched_triangles, mt_counts,
    counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
):
    # Outputs generated here are rectangular, because we are computing intersections between every ray
    # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
    ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
        working_start, working_hat, boundary_p, boundary_u, boundary_v, intersect_epsilon
    )
    valid, closest_trig, ray_r = _select_intersections(
        ray_r, trig_u, trig_v, valid, size_epsilon, ray_start_epsilon
    )

    # Can construct new_triangles now, just by looking at closest_trig and valid
    ray_triangle_matches = tf.where(valid, closest_trig, -tf.ones_like(valid, dtype=tf.int32))
    matched_triangles = tf.concat((matched_triangles, ray_triangle_matches), 0)
    mt_counts = tf.concat((mt_counts, tf.reshape(tf.shape(ray_triangle_matches)[0], (1,))), 0)
    intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)
    counter += 1

    # Select and compile the finished rays
    select_finished = tf.logical_and(valid, tf.equal(intersection_type, TARGET))
    finished_start = tf.concat((finished_start, tf.boolean_mask(working_start, select_finished)), axis=0)
    finished_hat = tf.concat((finished_hat, tf.boolean_mask(working_hat, select_finished)), axis=0)
    finished_meta = tf.concat((finished_meta, tf.boolean_mask(working_meta, select_finished)), axis=0)

    # Select working rays
    select_working = tf.logical_and(valid, tf.equal(intersection_type, OPTICAL))
    working_r = tf.reshape(tf.boolean_mask(ray_r, select_working), (-1, 1))
    working_start = tf.boolean_mask(working_start, select_working)
    working_hat = working_r * tf.boolean_mask(working_hat, select_working)
    working_n = tf.boolean_mask(working_n, select_working)
    working_meta = tf.boolean_mask(working_meta, select_working)

    # Gather the triangle data to match the working rays.
    closest_trig = tf.boolean_mask(closest_trig, select_working)
    selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
    mat_in = tf.gather(metadata_const[:, 0], closest_trig, axis=0)
    mat_out = tf.gather(metadata_const[:, 1], closest_trig, axis=0)

    # Determine the refractive index for each ray reaction
    n_in = tf.gather(working_n, mat_in, axis=1, batch_dims=1)
    n_out = tf.gather(working_n, mat_out, axis=1, batch_dims=1)

    # Perform the ray reactions.
    working_end = working_start + working_hat
    new_hat = snells_law_3d(
        working_hat, selected_boundary_norms, n_in, n_out, new_ray_length
    )

    return (
        finished_start, finished_hat, finished_meta,
        working_end, new_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms_const, metadata_const,
        matched_triangles, mt_counts,
        counter, trace_depth, intersect_epsilon, size_epsilon, ray_start_epsilon, new_ray_length
    )


@tf.function
def precompiled_trace_loop(
    ray_start, ray_hat, ray_n, ray_meta,
    boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
    triangle_map, triangle_map_indices,
    trace_depth, intersect_epsilon, new_ray_length
):
    finished_start = tf.zeros((0, 3), dtype=tf.float64)
    finished_hat = tf.zeros((0, 3), dtype=tf.float64)
    finished_meta = tf.zeros((0,), dtype=tf.int32)
    counter = tf.constant(0, dtype=tf.int32)

    (
        finished_start, finished_hat, finished_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
        triangle_map, triangle_map_indices,
        counter, trace_depth, intersect_epsilon, new_ray_length
    ) = tf.while_loop(
        _precompiled_while_cond,
        _precompiled_while_body,
        (
            finished_start, finished_hat, finished_meta,
            ray_start, ray_hat, ray_n, ray_meta,
            boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
            triangle_map, triangle_map_indices,
            counter, trace_depth, intersect_epsilon, new_ray_length
        ),
        shape_invariants=(
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 1)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, None)),
            tf.TensorShape((None, None)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None, 3)), tf.TensorShape((None, 3)),
            tf.TensorShape((None,)), tf.TensorShape((None,)),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(())
        )
    )

    return finished_start, finished_hat, finished_meta,


def _precompiled_while_cond(
        finished_start, finished_hat, finished_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p, boundary_u, boundary_v, boundary_norms, metadata,
        triangle_map, triangle_map_indices,
        counter, trace_depth, intersect_epsilon, new_ray_length
):
    return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_start)[0], 0))


def _precompiled_while_body(
        finished_start, finished_hat, finished_meta,
        working_start, working_hat, working_n, working_meta,
        boundary_p_const, boundary_u_const, boundary_v_const, boundary_norms_const, metadata_const,
        triangle_map_const, triangle_map_indices_const,
        counter, trace_depth, intersect_epsilon, new_ray_length
):
    # Process the triangle map.  It has -1 wherever rays do not intersect, so filter these ones out
    start_index, stop_index = triangle_map_indices_const[counter], triangle_map_indices_const[counter+1]
    gather_triangles = triangle_map_const[start_index:stop_index]

    valid = tf.not_equal(gather_triangles, -1)
    working_start = tf.boolean_mask(working_start, valid)
    working_hat = tf.boolean_mask(working_hat, valid)
    working_n = tf.boolean_mask(working_n, valid)
    working_meta = tf.boolean_mask(working_meta, valid)

    # Gather the triangle parts from the triangle map
    gather_triangles = tf.boolean_mask(gather_triangles, valid)
    boundary_p = tf.gather(boundary_p_const, gather_triangles)
    boundary_u = tf.gather(boundary_u_const, gather_triangles)
    boundary_v = tf.gather(boundary_v_const, gather_triangles)
    boundary_norms = tf.gather(boundary_norms_const, gather_triangles)
    metadata = tf.gather(metadata_const, gather_triangles)

    # Perform the intersection, as a raw intersection since we already know which triangle to intersect each
    # ray with.
    ray_r, trig_u, trig_v, valid = raw_line_triangle_intersection(
        working_start, working_hat, boundary_p, boundary_u, boundary_v, intersect_epsilon
    )

    # We already know every ray found a triangle, so project them all
    working_hat = tf.reshape(ray_r, (-1, 1)) * working_hat

    # Select and compile the finished rays
    intersection_type = metadata[:, 2]
    select_finished = tf.equal(intersection_type, TARGET)
    finished_start = tf.concat((finished_start, tf.boolean_mask(working_start, select_finished)), axis=0)
    finished_hat = tf.concat((finished_hat, tf.boolean_mask(working_hat, select_finished)), axis=0)
    finished_meta = tf.concat((finished_meta, tf.boolean_mask(working_meta, select_finished)), axis=0)

    # Select the active rays
    select_working = tf.equal(intersection_type, OPTICAL)
    working_start = tf.boolean_mask(working_start, select_working)
    working_hat = tf.boolean_mask(working_hat, select_working)
    working_n = tf.boolean_mask(working_n, select_working)
    working_meta = tf.boolean_mask(working_meta, select_working)

    metadata = tf.boolean_mask(metadata, select_working)
    boundary_norms = tf.boolean_mask(boundary_norms, select_working)

    # Determine the refractive index for each ray reaction
    n_in = tf.gather(working_n, metadata[:, 0], axis=1, batch_dims=1)
    n_out = tf.gather(working_n, metadata[:, 1], axis=1, batch_dims=1)

    # Perform the ray reactions.
    working_end = working_start + working_hat
    new_hat = snells_law_3d(
        working_hat, boundary_norms, n_in, n_out, new_ray_length
    )

    counter += 1
    return (
        finished_start, finished_hat, finished_meta,
        working_end, new_hat, working_n, working_meta,
        boundary_p_const, boundary_u_const, boundary_v_const, boundary_norms_const, metadata_const,
        triangle_map_const, triangle_map_indices_const,
        counter, trace_depth, intersect_epsilon, new_ray_length
    )
