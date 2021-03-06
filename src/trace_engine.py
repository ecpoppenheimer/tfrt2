from collections import namedtuple

import tensorflow as tf
import numpy as np


class TraceEngine3D:
    OPTICAL = 0
    STOP = 1
    TARGET = 2

    def __init__(self, materials):
        self.materials = materials
        self.active_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.finished_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.dead_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.stopped_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.all_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.unfinished_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.source_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.opticals = None
        self.stops = None
        self.targets = None
        self._new_ray_length = tf.constant(1.0, dtype=tf.float64)

        self.intersect_epsilon = 1e-10
        self.size_epsilon = 1e-10
        self.ray_start_epsilon = 1e-10

    def clear_rays(self, clear_sources=True):
        self.active_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.finished_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.dead_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.stopped_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.all_rays = tf.zeros((0, 7), dtype=tf.float64)
        self.unfinished_rays = tf.zeros((0, 7), dtype=tf.float64)
        if clear_sources:
            self.source_rays = tf.zeros((0, 7), dtype=tf.float64)

    def _fuse(self):
        """
        The first part of ray tracing, this function collects data from the parts and fuses it together into the
        correct format for the tracer.

        All boundaries will get converted into three chunks:
            boundary_points : tf.float64 tensor of shape (n, 9), where n is the number of faces, and the second
                dimension contains the coordinates of the three vertices, in order first, second, third, where
                the first point is the 'pivot' or center vertex of the triangle, and second - first and third - first
                form the two basis vectors in which the intersection will be computed.
            boundary_norms : tf.float64 tensor of shape (n, 3), where n is the number of faces and the second dimension
                contains the norm vector of each face.
            metadata : tf.int32 tensor of shape (n, 3), where n is the number of faces and the three elements in
                the second dimension are material_in, material_out (indices to the system materials list) and
                the last element identifies which kind of boundary this is: optical = 0, target = 1, stop = 2

        All sources will simply get concatenated together.  They form a chunk of data that is a tf.float64 tensor of
            shape (m, 7), where m is the number of rays and the second dimension contains the three coordinates of the
            ray start, the three coordinates of the ray end, and the wavelength.

        These four elements are returned in order
        """

        boundaries = self.opticals + self.stops + self.targets

        if boundaries:
            boundary_points = tf.concat([tf.concat(b.face_vertices, axis=1) for b in boundaries], axis=0)
            boundary_norms = tf.concat([b.norm for b in boundaries], axis=0)

            metadata = []
            for tp, box in zip(
                (TraceEngine3D.OPTICAL, TraceEngine3D.STOP, TraceEngine3D.TARGET),
                (self.opticals, self.stops, self.targets)
            ):
                if box:
                    for b in box:
                        metadata.append(np.broadcast_to(((b.mat_in, b.mat_out, tp),), b.norm.shape))
            metadata = tf.cast(tf.concat(metadata, axis=0), tf.int32)
        else:
            boundary_points = tf.zeros((0, 3), dtype=tf.float64)
            boundary_norms = tf.zeros((0, 3), dtype=tf.float64)
            metadata = tf.zeros((0, 3), dtype=tf.int32)

        return self.source_rays, boundary_points, boundary_norms, metadata

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

        self.active_rays, self.finished_rays, self.stopped_rays, self.dead_rays, self.unfinished_rays = \
            self._full_trace_loop(*self._fuse(), tf.convert_to_tensor(trace_depth, dtype=tf.int32))
        self.all_rays = tf.concat((
            self.active_rays, self.finished_rays, self.stopped_rays, self.dead_rays, self.unfinished_rays
        ), axis=0)

    @tf.function
    def _full_trace_loop(self, source_rays, boundary_points, boundary_norms, metadata, trace_depth):
        active_rays = tf.ensure_shape(tf.zeros((0, 7), dtype=tf.float64), (None, 7))
        finished_rays = tf.ensure_shape(tf.zeros((0, 7), dtype=tf.float64), (None, 7))
        dead_rays = tf.ensure_shape(tf.zeros((0, 7), dtype=tf.float64), (None, 7))
        stopped_rays = tf.ensure_shape(tf.zeros((0, 7), dtype=tf.float64), (None, 7))

        counter = tf.constant(0, dtype=tf.int32)
        (
            active_rays, finished_rays, dead_rays, stopped_rays, working_rays, boundary_points, boundary_norms,
            metadata, counter, trace_depth
        ) = tf.while_loop(
            self._full_while_cond,
            self._full_while_body,
            (
                active_rays, finished_rays, dead_rays, stopped_rays, source_rays, boundary_points, boundary_norms,
                metadata, counter, trace_depth
            ),
            shape_invariants=(
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 9)),
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 3)),
                tf.TensorShape(()),
                tf.TensorShape(())
            )
        )

        return active_rays, finished_rays, stopped_rays, dead_rays, working_rays

    def _full_while_cond(
        self, active_rays, finished_rays, dead_rays, stopped_rays, working_rays, boundary_points, boundary_norms,
        metadata, counter, trace_depth
    ):
        return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_rays)[0], 0))

    def _full_while_body(
        self, active_rays, finished_rays, dead_rays, stopped_rays, working_rays, boundary_points_const,
        boundary_norms_const, metadata_const, counter, trace_depth
    ):
        counter += 1
        # Outputs generated here are rectangular, because we are computing intersections between every ray
        # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
        ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
            working_rays, boundary_points_const, self.intersect_epsilon
        )
        valid, closest_trig, ray_r = self._select_intersections(
            ray_r, trig_u, trig_v, valid, self.size_epsilon, self.ray_start_epsilon
        )

        # At this point, active, finished, dead, and stopped rays contain no rays from this iteration;
        # working rays contains every ray traced this iteration, and valid, closest_trig, and the parametric solution
        # all match working rays;
        # and boundary_points, boundary_norms, and metadata are all organized per triangle.

        # Any ray that does not have a valid intersection is a dead ray, so filter them out
        dr = tf.boolean_mask(working_rays, tf.logical_not(valid))
        dead_rays = tf.concat((dead_rays, dr), axis=0)

        # Filter out dead rays and project all remaining rays into their intersection.
        working_rays = tf.boolean_mask(working_rays, valid)
        closest_trig = tf.boolean_mask(closest_trig, valid)
        ray_r = tf.reshape(tf.boolean_mask(ray_r, valid), (-1, 1))

        ray_starts = working_rays[:, :3]
        ray_ends = working_rays[:, 3:6]
        wavelengths = working_rays[:, 6]
        working_rays = tf.concat((
            ray_starts,
            ray_starts + ray_r * (ray_ends - ray_starts),
            tf.reshape(wavelengths, (-1, 1))
        ), axis=1)

        # Select the interaction type by gathering out of metadata.
        intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)

        # Sort the rays into their bucket
        select_stopped = tf.equal(intersection_type, self.STOP)
        stopped_rays = tf.concat((stopped_rays, tf.boolean_mask(working_rays, select_stopped)), axis=0)
        select_finished = tf.equal(intersection_type, self.TARGET)
        finished_rays = tf.concat((finished_rays, tf.boolean_mask(working_rays, select_finished)), axis=0)
        select_active = tf.equal(intersection_type, self.OPTICAL)
        working_rays = tf.boolean_mask(working_rays, select_active)
        active_rays = tf.concat((active_rays, working_rays), axis=0)

        # Filter all ray data down to just the active rays, and gather the triangle data to match.
        closest_trig = tf.boolean_mask(closest_trig, select_active)
        selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
        ref_index = tf.gather(metadata_const[:, :2], closest_trig, axis=0)

        # Determine the refractive index for each ray reaction
        ray_starts = working_rays[:, :3]
        ray_ends = working_rays[:, 3:6]
        wavelengths = working_rays[:, 6]
        all_ns_by_wavelength = tf.stack([material(wavelengths) for material in self.materials], axis=1)
        n_in = tf.gather(all_ns_by_wavelength, ref_index[:, 0], axis=1, batch_dims=1)
        n_out = tf.gather(all_ns_by_wavelength, ref_index[:, 1], axis=1, batch_dims=1)

        # Perform the ray reactions.
        working_rays = snells_law_3d(
            ray_starts, ray_ends, wavelengths, selected_boundary_norms, n_in, n_out, self._new_ray_length
        )

        return (
            active_rays, finished_rays, dead_rays, stopped_rays, working_rays, boundary_points_const,
            boundary_norms_const, metadata_const, counter, trace_depth
        )

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

        self.finished_rays = self._fast_trace_loop(*self._fuse(), tf.convert_to_tensor(trace_depth, dtype=tf.int32))

    @tf.function
    def _fast_trace_loop(self, source_rays, boundary_points, boundary_norms, metadata, trace_depth):

        finished_rays = tf.zeros((0, 7), dtype=tf.float64)
        counter = tf.constant(0, dtype=tf.int32)
        (
            finished_rays, working_rays, boundary_points, boundary_norms, metadata, counter, trace_depth
        ) = tf.while_loop(
            self._fast_while_cond,
            self._fast_while_body,
            (
                finished_rays, source_rays, boundary_points, boundary_norms, metadata, counter, trace_depth
            ),
            shape_invariants=(
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 9)),
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 3)),
                tf.TensorShape(()),
                tf.TensorShape(())
            )
        )

        return finished_rays

    def _fast_while_cond(
            self, finished_rays, working_rays, boundary_points, boundary_norms, metadata, counter, trace_depth
    ):
        return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_rays)[0], 0))

    def _fast_while_body(
            self, finished_rays, working_rays, boundary_points_const, boundary_norms_const, metadata_const,
            counter, trace_depth
    ):
        counter += 1
        # Outputs generated here are rectangular, because we are computing intersections between every ray
        # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
        ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
            working_rays, boundary_points_const, self.intersect_epsilon
        )
        valid, closest_trig, ray_r = self._select_intersections(
            ray_r, trig_u, trig_v, valid, self.size_epsilon, self.ray_start_epsilon
        )

        # Select and compile the finished rays
        intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)
        select_finished = tf.logical_and(valid, tf.equal(intersection_type, self.TARGET))
        fin_r = tf.boolean_mask(working_rays, select_finished)
        fin_r_r = tf.reshape(tf.boolean_mask(ray_r, select_finished), (-1, 1))
        finished_starts = fin_r[:, :3]
        finished_ends = fin_r[:, 3:6]
        finished_wv = fin_r[:, 6]
        new_finished_rays = tf.concat((
            finished_starts,
            finished_starts + fin_r_r * (finished_ends - finished_starts),
            tf.reshape(finished_wv, (-1, 1))
        ), axis=1)
        finished_rays = tf.concat((finished_rays, new_finished_rays), axis=0)

        # Select the working rays
        select_working = tf.logical_and(valid, tf.equal(intersection_type, self.OPTICAL))
        working_rays = tf.boolean_mask(working_rays, select_working)
        ray_r = tf.reshape(tf.boolean_mask(ray_r, select_working), (-1, 1))

        # Gather the triangle data to match the working rays.
        closest_trig = tf.boolean_mask(closest_trig, select_working)
        selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
        ref_index = tf.gather(metadata_const[:, :2], closest_trig, axis=0)

        ray_starts = working_rays[:, :3]
        ray_ends = ray_starts + ray_r * (working_rays[:, 3:6] - ray_starts)
        wavelengths = working_rays[:, 6]

        # Determine the refractive index for each ray reaction
        all_ns_by_wavelength = tf.stack([material(wavelengths) for material in self.materials], axis=1)
        n_in = tf.gather(all_ns_by_wavelength, ref_index[:, 0], axis=1, batch_dims=1)
        n_out = tf.gather(all_ns_by_wavelength, ref_index[:, 1], axis=1, batch_dims=1)

        # Perform the ray reactions.
        working_rays = snells_law_3d(
            ray_starts, ray_ends, wavelengths, selected_boundary_norms, n_in, n_out, self._new_ray_length
        )

        return (
            finished_rays, working_rays, boundary_points_const, boundary_norms_const, metadata_const, counter,
            trace_depth
        )

    def get_trace_samples(self, trace_depth):
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

        In order to accommodate parallelization across systems, this function WILL CALL SYSTEM_UPDATE(), unlike
        ray_trace and fast_trace.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays
        """
        self.update()
        params = (c.parameters.numpy() for c in self.parametric_optics)
        trig_map, trig_map_indices = self._sample_loop(
            *self._fuse(), tf.convert_to_tensor(trace_depth, dtype=tf.int32)
        )
        return params, self.source_rays.numpy(), (trig_map.numpy(), trig_map_indices.numpy())

    @tf.function
    def _sample_loop(self, source_rays, boundary_points, boundary_norms, metadata, trace_depth):
        matched_triangles = tf.zeros((0,), dtype=tf.int32)
        mt_counts = tf.zeros((0,), dtype=tf.int32)
        counter = tf.constant(0, dtype=tf.int32)
        (
            matched_triangles, mt_counts, working_rays, boundary_points, boundary_norms, metadata, counter, trace_depth
        ) = tf.while_loop(
            self._sample_while_cond,
            self._sample_while_body,
            (
                matched_triangles, mt_counts, source_rays, boundary_points, boundary_norms, metadata, counter,
                trace_depth
            ),
            shape_invariants=(
                tf.TensorShape((None,)),
                tf.TensorShape((None,)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 9)),
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 3)),
                tf.TensorShape(()),
                tf.TensorShape(())
            )
        )

        # These changes will turn counts into a more convenient form to slice out the needed data later:
        # to get layer n, slice from mt_counts[n] to mt_counts[n+1]
        mt_counts = tf.pad(mt_counts, ((1, 0),))
        mt_counts = tf.cumsum(mt_counts)
        return matched_triangles, mt_counts

    def _sample_while_cond(
        self, matched_triangles, mt_counts, working_rays, boundary_points, boundary_norms, metadata, counter,
        trace_depth
    ):
        return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_rays)[0], 0))

    def _sample_while_body(
        self, matched_triangles, mt_counts, working_rays, boundary_points_const, boundary_norms_const,
        metadata_const, counter, trace_depth
    ):
        # Outputs generated here are rectangular, because we are computing intersections between every ray
        # with every boundary.  First dimension indexes rays, second dimension indexes boundaries.
        ray_r, trig_u, trig_v, valid = full_line_triangle_intersection(
            working_rays, boundary_points_const, self.intersect_epsilon
        )
        valid, closest_trig, ray_r = self._select_intersections(
            ray_r, trig_u, trig_v, valid, self.size_epsilon, self.ray_start_epsilon
        )

        # Can construct new_triangles now, just by looking at closest_trig and valid
        ray_triangle_matches = tf.where(valid, closest_trig, -tf.ones_like(valid, dtype=tf.int32))
        matched_triangles = tf.concat((matched_triangles, ray_triangle_matches), 0)
        mt_counts = tf.concat((mt_counts, tf.reshape(tf.shape(ray_triangle_matches)[0], (1,))), 0)
        counter += 1

        # Select working rays
        intersection_type = tf.gather(metadata_const[:, 2], closest_trig, axis=0)
        select_working = tf.logical_and(valid, tf.equal(intersection_type, self.OPTICAL))
        working_rays = tf.boolean_mask(working_rays, select_working)
        ray_r = tf.reshape(tf.boolean_mask(ray_r, select_working), (-1, 1))

        # Gather the triangle data to match the working rays.
        closest_trig = tf.boolean_mask(closest_trig, select_working)
        selected_boundary_norms = tf.gather(boundary_norms_const, closest_trig, axis=0)
        ref_index = tf.gather(metadata_const[:, :2], closest_trig, axis=0)

        ray_starts = working_rays[:, :3]
        ray_ends = ray_starts + ray_r * (working_rays[:, 3:6] - ray_starts)
        wavelengths = working_rays[:, 6]

        # Determine the refractive index for each ray reaction
        all_ns_by_wavelength = tf.stack([material(wavelengths) for material in self.materials], axis=1)
        n_in = tf.gather(all_ns_by_wavelength, ref_index[:, 0], axis=1, batch_dims=1)
        n_out = tf.gather(all_ns_by_wavelength, ref_index[:, 1], axis=1, batch_dims=1)

        # Perform the ray reactions.
        working_rays = snells_law_3d(
            ray_starts, ray_ends, wavelengths, selected_boundary_norms, n_in, n_out, self._new_ray_length
        )

        return (
            matched_triangles, mt_counts, working_rays, boundary_points_const, boundary_norms_const, metadata_const,
            counter, trace_depth
        )

    def precompiled_trace(self, trace_depth, sample):
        """
        Perform a trace from a precompiled sample, only tracking finished rays.

        This function is designed to be used with backpropigation; it requires a precompilation step
        (done with get_trace_samples() ) to determine which boundary each ray will intersect with at each step.
        That this function does not have to intersect every ray with every boundary means it is vastly more memory
        efficient.

        system.update() should NOT be called before this - this class will do that itself.  Please note that unlike
        other ray tracing routines, this function will CHANGE THE PARAMETERS IN PARAMETRIC OPTICS (and source rays)
        to the values they had during precompilation.  I still have not decided how to handle the parameters during
        backprop, but a master parameter set is most likely going to have to be used.

        Parameters
        ----------
        trace_depth : int
            The maximum number of tracing iterations to perform.  Rays that have not finished by this many cycles will
            get added to unfinished_rays
        sample : tuple
            A tuple as produced by get_trace_samples.  The elements are:
            params : tuple
                A tuple of parameter values.  Needed because I am planning for this operation
                to be run asynchronously with sample generation.  Parametric optics will be set with these parameter
                values before the trace is run, to guarantee that the trace generated here is identical to that
                used to generate the sample.
            source_rays : np.float64 ndarray with shape (None, 7)
                The source ray data to use for this trace
            triangles : tf.int32 tf.RaggedTensor
                A ragged tensor whose first dimension indexes trace iterations that contain at least one working ray
                and whose second dimension maps working rays to boundary triangles.
        """
        params, source_rays, triangles = sample
        self.update_params_from(params)
        self.source_rays = source_rays
        self.finished_rays = self._precompiled_trace_loop(
            *self._fuse(),
            *triangles,
            tf.convert_to_tensor(trace_depth, dtype=tf.int32)
        )

    @tf.function
    def _precompiled_trace_loop(
        self, source_rays, boundary_points, boundary_norms, metadata, triangle_map, triangle_map_indices, trace_depth
    ):
        finished_rays = tf.zeros((0, 7), dtype=tf.float64)
        counter = tf.constant(0, dtype=tf.int32)

        (
            finished_rays, working_rays, boundary_points, boundary_norms, metadata, triangle_map, triangle_map_indices,
            counter, trace_depth
        ) = tf.while_loop(
            self._precompiled_while_cond,
            self._precompiled_while_body,
            (
                finished_rays, source_rays, boundary_points, boundary_norms, metadata, triangle_map,
                triangle_map_indices, counter, trace_depth
            ),
            shape_invariants=(
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 7)),
                tf.TensorShape((None, 9)),
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 3)),
                tf.TensorShape((None,)),
                tf.TensorShape((None,)),
                tf.TensorShape(()),
                tf.TensorShape(())
            )
        )

        return finished_rays

    def _precompiled_while_cond(
        self, finished_rays, working_rays, boundary_points, boundary_norms, metadata, triangle_map,
        triangle_map_indices, counter, trace_depth
    ):
        return tf.logical_and(tf.less(counter, trace_depth), tf.greater(tf.shape(working_rays)[0], 0))

    def _precompiled_while_body(
        self, finished_rays, working_rays, boundary_points_const, boundary_norms_const, metadata_const,
        triangle_map_const, triangle_map_indices_const, counter, trace_depth
    ):
        # Process the triangle map.  It has -1 wherever rays do not intersect, so filter these ones out
        start_index, stop_index = triangle_map_indices_const[counter], triangle_map_indices_const[counter+1]
        gather_triangles = triangle_map_const[start_index:stop_index]

        valid = tf.not_equal(gather_triangles, -1)
        gather_triangles = tf.boolean_mask(gather_triangles, valid)
        working_rays = tf.boolean_mask(working_rays, valid)

        # Gather the triangle parts from the triangle map
        boundary_points = tf.gather(boundary_points_const, gather_triangles)
        boundary_norms = tf.gather(boundary_norms_const, gather_triangles)
        metadata = tf.gather(metadata_const, gather_triangles)

        # Perform the intersection, as a raw intersection since we already know which triangle to intersect each
        # ray with.
        ray_r, trig_u, trig_v, valid = raw_line_triangle_intersection(
            working_rays, boundary_points, self.intersect_epsilon
        )

        # We already know every ray found a triangle, so project them all
        ray_starts = working_rays[:, :3]
        ray_ends = working_rays[:, 3:6]
        wavelengths = working_rays[:, 6]
        working_rays = tf.concat((
            ray_starts,
            ray_starts + tf.reshape(ray_r, (-1, 1)) * (ray_ends - ray_starts),
            tf.reshape(wavelengths, (-1, 1))
        ), axis=1)

        # Select and compile the finished rays
        intersection_type = metadata[:, 2]
        select_finished = tf.equal(intersection_type, self.TARGET)
        finished_rays = tf.concat((finished_rays, tf.boolean_mask(working_rays, select_finished)), axis=0)

        # Select the active rays
        select_working = tf.equal(intersection_type, self.OPTICAL)
        working_rays = tf.boolean_mask(working_rays, select_working)
        metadata = tf.boolean_mask(metadata, select_working)
        boundary_norms = tf.boolean_mask(boundary_norms, select_working)
        ray_starts = working_rays[:, :3]
        ray_ends = working_rays[:, 3:6]
        wavelengths = working_rays[:, 6]

        # Determine the refractive index for each ray reaction
        all_ns_by_wavelength = tf.stack([material(wavelengths) for material in self.materials], axis=1)
        n_in = tf.gather(all_ns_by_wavelength, metadata[:, 1], axis=1, batch_dims=1)
        n_out = tf.gather(all_ns_by_wavelength, metadata[:, 2], axis=1, batch_dims=1)

        # Perform the ray reactions.
        working_rays = snells_law_3d(
            ray_starts, ray_ends, wavelengths, boundary_norms, n_in, n_out,
            self._new_ray_length
        )

        counter += 1
        return (
            finished_rays, working_rays, boundary_points_const, boundary_norms_const, metadata_const,
            triangle_map_const, triangle_map_indices_const, counter, trace_depth
        )

    def update_params_from(self, param_tuple):
        for component, value in zip(self.parametric_optics, param_tuple):
            component.param_assign(value)
            component.update()

    @staticmethod
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

    @property
    def new_ray_length(self):
        return self._new_ray_length

    @new_ray_length.setter
    def new_ray_length(self, val):
        self._new_ray_length = tf.constant(val, dtype=tf.float64)


@tf.function(input_signature=(
        tf.TensorSpec(shape=(None, 7), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 9), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64)
))
def full_line_triangle_intersection(working_rays, boundary_points, epsilon):
    # Expand / meshgrid the ray and boundary data.  First dimension indexes rays, second dimension indexes boundaries.
    ray_count = tf.shape(working_rays)[0]
    boundary_count = tf.shape(boundary_points)[0]
    working_rays = tf.expand_dims(working_rays, 1)
    working_rays = tf.broadcast_to(working_rays, (ray_count, boundary_count, 7))
    boundary_points = tf.expand_dims(boundary_points, 0)
    boundary_points = tf.broadcast_to(boundary_points, (ray_count, boundary_count, 9))

    return raw_line_triangle_intersection(working_rays, boundary_points, epsilon)


@tf.function(input_signature=(
        tf.TensorSpec(shape=(None, 7), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 9), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64)
))
def line_triangle_intersection(working_rays, boundary_points, epsilon):
    """
    This stub of a function exists solely so I can ensure that raw_line_triangle_intersection is always wrapped
    in a tf.function with an input signature.  Difficult because it gets rank 3 tensors from
    full_line_triangle_intersection and rank 2 tensors from here.
    """
    return raw_line_triangle_intersection(working_rays, boundary_points, epsilon)


def raw_line_triangle_intersection(rays, boundaries, epsilon):
    """
    Low-level function that actually computes the intersection.  Algebra was done on wolfram alpha, so unfortunately
    not particularly readable, but it is just a solution to the parametric equation defined by
    ray = ray_start + r * (ray_end - ray_start)
    boundary = boundary_1 + u * (boundary_2 - boundary_1) + v * (boundary_3 - boundary_1)

    But I am copying the math over from tfrt1, which had the naming convention boundary_1 = p, boundary_2 = 1,
    boundary_3 = 2
    """

    rx1, ry1, rz1, rx2, ry2, rz2, _ = tf.unstack(rays, axis=-1)
    xp, yp, zp, x1, y1, z1, x2, y2, z2 = tf.unstack(boundaries, axis=-1)

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
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
))
def snells_law_3d(ray_starts, ray_ends, wavelengths, boundary_norms, n_in, n_out, new_ray_length):
    # A vector representing the ray direction.
    u = tf.math.l2_normalize(ray_ends - ray_starts, axis=1)

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

    new_end = ray_ends + new_ray_length * new_vector
    return tf.concat((ray_ends, new_end, tf.reshape(wavelengths, (-1, 1))), axis=1)
