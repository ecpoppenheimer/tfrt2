import traceback
import pickle
import pathlib
import abc

import scipy.interpolate
import tensorflow as tf
import pyvista as pv
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import scipy

import tfrt2.mesh_tools as mt
import tfrt2.component_widgets as cw
import tfrt2.drawing as drawing
from tfrt2.settings import Settings
from tfrt2.sources import get_rotation_quaternion_from_u_to_v
from tfrt2.vector_generator import FromVectorVG


class TriangleOptic:
    """
    Basic 3D optical element composed of a triangulated mesh.

    Please note that vertices and faces are always assumed to be tensorflow tensors.
    """
    dimension = 3
    BASE_ORIENTATION = np.array((0.0, 0.0, 1.0), dtype=np.float64)

    def __init__(
        self, driver, system_path, input_settings, mesh=None, mat_in=None, mat_out=None, hold_load=False,
        suppress_ui=False, flip_norm=False, active=True
    ):
        """
        Basic 3D Triangulated optical surface.

        This class is suitable for non-parametric optics and technical surfaces, like stops or targets.

        It is the user's responsibility to feed and initialize the settings given to this object.  Omitting certain
        settings will suppress behavior in the component controller (can omit displaying IO for a technical surface,
        for instance).  Defaults will only be established by this class if explicitly stated in this documentation
        (though please note that controllers can also establish defaults).

        In order to be compatible with the drawer and tracer, the following operations must be completed whenever the
        mesh is updated.  These operations are not performed automatically because different more complicated
        derivatives might need to do extra operations in between
        1) Set the faces (usually done in from_mesh or some kind of remesh operation).
        2) Set the vertices (The base class does it in from_mesh, but a parametric class will do this in update()).
        3) Separate the vertices out by face by calling call gather_faces().  THIS MUST BE EXPLICITLY DONE IN UPDATE BY
            CHILD CLASSES.
        4) Compute the norm by calling compute_norm().  THIS MUST BE EXPLICITLY DONE IN UPDATE BY CHILD CLASSES.

        Parameters
        ----------
        name : str
            A unique name used to identify this component.  Uniqueness will be enforced when this component is
            added to a system.
        driver : client.OpticClientWindow
            The top level client window.
        settings : dict or settings.Settings
            The settings for this optical element.
        mesh : pv.PolyData, optional
            If specified, this optic will be driven by the mesh and an I/O controller will not be generated.
            If not specified, this optic will be driven by an I/O controller, and it is the responsibility of the user
            to ensure that settings are fed that establish a default 'mesh_input_path'.
        mat_in : int
            The index of the material to use for the inside of this optic (area pointing away from norm).
        mat_out : int
            The index of the material to use for the outside of this optic (area pointing toward from norm).

        Recognized Settings
        -------------------
        visible : bool
            Whether the optic is visible in the 3D display.  No effect on tracing.
        color : str
            The pyvista color string used to draw the optic in the 3D display.  No effect on tracing.
        show_edges : bool
            Whether or not to draw lines along the edges of the faces in the 3D display.
        mesh_output_path : str
            The path where to save this mesh.
        mesh_input_path : str
            The path from which to load this mesh.  This path will be used to initialize the surface if mesh
            isn't specified.

        """
        self.settings = input_settings or Settings()
        self.name = None
        self.mat_in = mat_in or 0
        self.mat_out = mat_out or 0
        self.vertices = None
        self._raw_faces = None
        self.driver = driver
        self.base_mesh = None
        self.tiled_base_mesh = None
        self.face_vertices = None
        self.norm = None
        self.p = None
        self.u = None
        self.v = None
        self.zero_points = None
        self.drawer = None
        self.settings.establish_defaults(
            frozen=False,
            flip_norm=flip_norm,
            active=active,
            translation=np.array((0.0, 0.0, 0.0), dtype=np.float64),
            rotation=TriangleOptic.BASE_ORIENTATION.copy(),
            tileable=False,
            tile_1_low_count=0,
            tile_1_high_count=0,
            tile_1_vector=np.array((1.0, 0.0, 0.0)),
            tile_2_low_count=0,
            tile_2_high_count=0,
            tile_2_vector=np.array((0.0, 1.0, 0.0))
        )
        self.translation = None
        self.rotation = None
        self.tcp_ack_remeshed = False
        self.p_controller_ack_remeshed = False

        if not hold_load:
            if mesh is None:
                self.load(self.settings.mesh_input_path)
            else:
                self.from_mesh(mesh)

        self.controller_widgets = []
        if self.driver.driver_type == "client" and not suppress_ui:
            self.controller_widgets.append(OpticController(self, driver, system_path))

        self.update_rotation(False)
        self.update_translation(False)

    def load(self, in_path):
        self.from_mesh(pv.read(in_path))

    def save(self, out_path):
        self.as_mesh().save(out_path)

    def as_mesh(self):
        try:
            return pv.PolyData(self.vertices.numpy(), mt.pack_faces(self.faces))
        except Exception:
            return pv.PolyData(np.array(self.vertices), mt.pack_faces(np.array(self.faces)))

    def from_mesh(self, mesh):
        self.base_mesh = mesh
        vertices, faces = mesh.points, mt.unpack_faces(mesh.faces)
        tiling_params = self._compute_tiling(vertices, faces)
        self.zero_points = self._perform_vertex_tiling(vertices, tiling_params)
        self._raw_faces = self._perform_face_tiling(faces, tiling_params)
        self.tiled_base_mesh = pv.PolyData(self.zero_points, mt.pack_faces(self._raw_faces))
        self.update()

    def gather_faces(self):
        self.face_vertices = mt.points_from_faces(self.vertices, self.faces)

    def compute_norm(self):
        if self.settings.active:
            self.p, second_points, third_points = self.face_vertices
            self.u = second_points - self.p
            self.v = third_points - self.p
            self.norm = tf.linalg.normalize(tf.linalg.cross(self.u, self.v), axis=1)[0]
        else:
            self.p = tf.zeros((0, 3), dtype=tf.float64)
            self.u = tf.zeros((0, 3), dtype=tf.float64)
            self.v = tf.zeros((0, 3), dtype=tf.float64)
            self.norm = tf.ones((0, 3), dtype=tf.float64)

    def update(self):
        if self.translation is not None and self.rotation is not None:
            rotated_vertices = tf.convert_to_tensor(self.rotation.rotate(self.zero_points), dtype=tf.float64)
            self.vertices = rotated_vertices + self.translation
        else:
            self.vertices = tf.convert_to_tensor(self.zero_points, dtype=tf.float64)

        # Separate the vertices out by face
        self.gather_faces()
        self.compute_norm()

    def update_tiling(self):
        """
        Expand vertices and faces to tile the optic.
        """
        self.from_mesh(self.base_mesh)
        self.update_and_redraw()

    def _compute_tiling(self, vertices, faces):
        """
        Returns None if no tiling needs to happen.  Otherwise, returns a tuple of:
        (total_repeats, base_vertex_count, base_face_count, combined_offsets)
        These are the parameters that need to be fed to the tiling sub functions to tile various optic components.
        """
        if self.settings.tileable:
            tile_1_offsets = \
                np.arange(self.settings.tile_1_low_count, self.settings.tile_1_high_count+1)
            tile_2_offsets = \
                np.arange(self.settings.tile_2_low_count, self.settings.tile_2_high_count+1)
            total_repeats = len(tile_1_offsets) * len(tile_2_offsets)
            if total_repeats == 1:
                return None
            base_vertex_count = len(vertices)
            base_face_count = len(faces)

            tile_1_offsets, tile_2_offsets = np.meshgrid(tile_1_offsets, tile_2_offsets)
            tile_1_offsets = np.expand_dims(tile_1_offsets.flatten(), axis=1)
            tile_2_offsets = np.expand_dims(tile_2_offsets.flatten(), axis=1)
            combined_offsets = tile_1_offsets * np.expand_dims(self.settings.tile_1_vector, axis=0) + \
                tile_2_offsets * np.expand_dims(self.settings.tile_2_vector, axis=0)

            return total_repeats, base_vertex_count, base_face_count, combined_offsets
        else:
            return None

    @staticmethod
    def _perform_vertex_tiling(vertices, tiling_params):
        """
        Tile a set of vertices with the tiling parameters.  Feed the return from _compute_tiling directly
        to this function.  If there is no tiling to perform, this will return vertices as is, so this function can be
        used inline without any case checking.
        """
        if tiling_params is None:
            return vertices
        else:
            total_repeats, base_vertex_count, base_face_count, combined_offsets = tiling_params
            # For vertices, need to tile the vertices to the repeats, and add the offsets (which are the repeats)
            # to the vertices.
            return np.tile(vertices, (total_repeats, 1)) + \
                np.repeat(combined_offsets, base_vertex_count, axis=0)

    @staticmethod
    def _perform_face_tiling(faces, tiling_params):
        """
        Tile a set of faces with the tiling parameters in parameters.  Feed the return from _compute_tiling directly
        to this function.  If there is no tiling to perform, this will return faces as is, so this function can be
        used inline without any case checking.
        """
        if tiling_params is None:
            return faces
        else:
            total_repeats, base_vertex_count, base_face_count, combined_offsets = tiling_params
            # For faces, need to tile raw_faces for the total repeats, and add an offset derived from the
            # number of base vertices.
            # The second line creates (0, 0, 0, 0, ..., 1, 1, 1, 1, ..., 2, 2, 2, 2, ..., ...) with total count equal
            # to base_face_count*total_repeats.  Then multiplies by the vertex count, and finally gets an extra
            # dimension added to match the 2D shape of the faces.
            return np.tile(faces, (total_repeats, 1)) +\
                np.expand_dims(np.repeat(np.arange(total_repeats), base_face_count) * base_vertex_count, axis=1)

    def redraw(self):
        """
        Only works if self.drawer was set, which must be done manually
        """
        if self.drawer is not None:
            self.drawer.draw()

    def update_and_redraw(self):
        self.update()
        self.redraw()

    def update_rotation(self, do_update=True):
        try:
            self.rotation = get_rotation_quaternion_from_u_to_v(
                TriangleOptic.BASE_ORIENTATION, np.array(self.settings.rotation, dtype=np.float64)
            )
            if do_update:
                self.update()
                self.redraw()
        except KeyError:
            pass

    def update_translation(self, do_update=True):
        try:
            self.translation = tf.convert_to_tensor(self.settings.translation, dtype=tf.float64)
            if do_update:
                self.update()
                self.redraw()
        except KeyError:
            pass

    @property
    def faces(self):
        if self.settings.flip_norm:
            return np.take(self._raw_faces, [2, 1, 0], axis=1)
        else:
            return self._raw_faces


class ParametricTriangleOptic(TriangleOptic):
    """
    3D Triangulated parametric optic, with advanced features for use with gradient descent optimization.

    Update adds some extra steps beyond the canonical ones for TriangleOptic:
    1) Apply constraints.
    2) Set the vertices from the zero points, vectors, and parameters, via get_vertices_from_params()
    3) Separate the vertices out by face via gather_faces()
    4) Apply the vertex update map via try_vum()
    5) Compute the norm via compute_norm().
    """
    def __init__(
        self,
        driver,
        system_path,
        settings,
        vector_generator,
        mesh=None,
        mat_in=None,
        mat_out=None,
        enable_vum=True,
        enable_accumulator=True,
        filter_fixed=None,
        filter_drivers=None,
        attach_to_driver=None,
        qt_compat=False,
        flip_norm=False,
        constraints=None
    ):
        """
        Optic whose vertices are parameter-driven.

        This optic has a public interface that allows to classify vertices into three distinct sets: fixed vertices,
        that do not move at all as parameters are changed, driver vertices, which have their own dedicated parameter,
        and driven vertices, which are controlled by another vertex's parameter.  This system allows for a rich
        collection of constraints to be applied to the optic, in a more elegant way than using the constraints list,
        since using these sets allows the underlying parameter space to be smaller.

        Vertices are classified when the optic is generated from a mesh, in the from_mesh function, which will be
        called by the constructor.  The sets are mutually exclusive: fixed vertices are classified
        first, and any vertex not classified as fixed passes to the next round.  Driver vertices are classified next.
        Any vertex not yet classified into either set will then attempt to get attached to a driver vertex.  Finally,
        any vertex that did not get attached to a driver will be added to fixed.

        Vertex classification is done by a function, which receives as a parameter a set of points (a numpy array of
        shape (n, 3) ) and returns a numpy boolean array of shape (n,) stating whether or not each point got classified
        into this set.  These functions can be specified either by specifying them as keyword arguments to the
        constructor, or by subclassing and implementing them as methods.  The descriptions of these functions are
        in the constructor documentation below.

        It is the user's responsibility to feed and initialize the settings given to this object.  Omitting certain
        settings will suppress behavior in the component controller (can omit displaying IO for a technical surface,
        for instance).  Defaults will only be established by this class if explicitly stated in this documentation
        (though please note that controllers can also establish defaults)

        Parameters
        ----------
        vector_generator : callable
            A function that will generate a set of n vectors when given a set of n 3D points.  Used to attach
            vectors to each vertex so they can be moved by the parameters.
        driver : client.OpticClientWindow
            The top level client window.
        settings : dict or settings.Settings
            The settings for this optical element.
        mesh : pv.PolyData, optional
            If specified, this optic will be driven by the mesh and an I/O controller will not be generated.
            If not specified, this optic will be driven by an I/O controller, and it is the responsibility of the user
            to ensure that settings are fed that establish a default 'mesh_input_path'.
        mat_in : int
            The index of the material to use for the inside of this optic (area pointing away from norm).
        mat_out : int
            The index of the material to use for the outside of this optic (area pointing toward from norm).
        filter_fixed : callable, optional
            Function that when given an array of vertices, returns a boolean array with one element for each vertex
            denoting whether this vertex should be fixed or not.  Will have NO EFFECT if a subclass implements a method
            with the same name.
        filter_drivers : callable, optional
            Function that when given an array of vertices, returns a boolean array with one element for each vertex
            denoting whether this vertex should be a driver vertex or not.  Will have NO EFFECT if a subclass implements
            a method with the same name.
        attach_to_driver : callable, optional
            A function that is given a single vertex, and an array of vertices, and returns a boolean array labeling
            vertices that should be driven by this one.  Each driven vertex can only ever be attached to a single
            driver, but this task is accomplished by this class, so the user does not need to take care to filter
            out duplicate assignments.
            vertex in the array that should be attached to and driven by the given vertex.  Will have NO EFFECT if a
            subclass implements a method with the same name.
        qt_compat : bool, optional
            Defaults to False.  If true, adds pyqt signals to this object, which UI elements can respond to.

        Public Attributes
        -----------------
        constraints : list of callables
            Each callable in this list will be called with the optic as the only argument, at the beginning of update().
            This is intended to be used to ensure various conditions and constraints are applied to the parameters, but
            it can be used more broadly to perform pre-update tasks, if needed.

        Recognized Settings
        -------------------
        All settings from TriangleOptic
        parameters_path : str
            The path where to store and load the parameters
        norm_arrow_visibility : bool
            Whether to draw norm arrows in the 3D display.
        norm_arrow_length : float
            The length of the norm arrows drawn in the 3D display.
        norm_arrow_visibility : bool
            Whether to draw parameter arrows in the 3D display.
        norm_arrow_length : float
            The length of the parameter arrows drawn in the 3D display.
        vum_active : bool
            Whether to use the vertex update map.  The option to set this only appears in the client if
            the vum is enabled in code for a given instance.
        accumulator_active : bool
            Whether to use the accumulator.  The option to set this only appears in the client if
            the vum is enabled in code for a given instance.
        mt_origin : 3-tuple of floats
            The coordinates of the origin to use when calculating a vum or accumulator.
        mt_origin_visible : bool
            Whether the origin is visible in the 3D display

        """
        if not hasattr(self, "filter_fixed"):
            if filter_fixed is not None:
                self.filter_fixed = filter_fixed
        if not hasattr(self, "filter_drivers"):
            if filter_drivers is not None:
                self.filter_drivers = filter_drivers
        if not hasattr(self, "attach_to_driver"):
            if attach_to_driver is not None:
                self.attach_to_driver = attach_to_driver

        self.enable_vum = enable_vum
        self.vum = None
        self.enable_accumulator = enable_accumulator
        self.accumulator = None
        self.vector_generator = vector_generator
        self.parameters = None
        self.zero_points = None
        self.vectors = None
        self.fixed_points = None
        self.gather_vertices = None
        self.gather_params = None
        self.initials = None
        self.any_fixed = False
        self.any_driven = False
        self.driver_indices = np.zeros((0,), dtype=np.int32)
        self.driven_indices = np.zeros((0,), dtype=np.int32)
        self.fixed_indices = np.zeros((0,), dtype=np.int32)
        self.v_to_p = None
        self.full_params = False
        self._qt_compat = qt_compat
        self._d_mat = None
        self.movable_indices = None
        self.smoother = None
        self.base_mesh = None
        if qt_compat:
            self.parameters_updated = ParamsUpdatedSignal()
        else:
            self.parameters_updated = None

        super().__init__(
            driver, system_path, settings, None, mat_in=mat_in, mat_out=mat_out, hold_load=True, flip_norm=flip_norm
        )

        self.settings.establish_defaults(
            vum_origin=(0, 0, 0),
            vum_active=True,
            accumulator_origin=(0, 0, 0),
            accumulator_active=True,
            smooth_stddev=.05,
            smooth_active=True,
            relative_lr=1.0,
        )

        if (self.enable_vum or self.enable_accumulator) and self.driver.driver_type == "client":
            self.mt_controller = MeshTricksController(self, driver)
            self.controller_widgets.append(self.mt_controller)
        else:
            self.mt_controller = None

        self.constraints = self._process_constraints(constraints)

        # Having some difficulty with initialization order, so need to hold off on loading until now
        if mesh is None:
            self.load(self.settings.mesh_input_path)
        else:
            self.from_mesh(mesh)

    def from_mesh(self, mesh, initials=None):
        """
        If initials is not None, it must be either a scalar, or it must be shaped like the parameters (taking into
        account the fixed / driven vertices), or it must be shaped like the vertices (or at least, have one element
        per vertex).
        """
        self.base_mesh = mesh.copy()
        all_indices = np.arange(mesh.points.shape[0])
        available_indices = all_indices.copy()
        drivens_driver = {}
        driven_indices = set()

        if hasattr(self, "filter_fixed"):
            fixed_mask = self.filter_fixed(mesh.points)
            fixed_indices = available_indices[fixed_mask]
            available_indices = available_indices[np.logical_not(fixed_mask)]
        else:
            fixed_indices = []

        if hasattr(self, "filter_drivers") and hasattr(self, "attach_to_driver"):
            driver_mask = self.filter_drivers(mesh.points[available_indices])
            driver_indices = available_indices[driver_mask]
            available_indices = available_indices[np.logical_not(driver_mask)]

            available_indices = set(available_indices)
            for driver in driver_indices:
                attached_indices = all_indices[self.attach_to_driver(mesh.points[driver], mesh.points)]
                attached_indices = set(attached_indices) & available_indices
                available_indices -= attached_indices
                driven_indices |= attached_indices
                for each in attached_indices:
                    drivens_driver[each] = driver

            # any indices still in available_indices are left over, and need to be added to driver_indices
            driver_indices = np.concatenate((driver_indices, np.array(list(available_indices), dtype=np.int32)))

            # Add each driver as a driver to itself into drivens_driver.
            for driver in driver_indices:
                drivens_driver[driver] = driver
        else:
            driver_indices = available_indices

        # At this point we should be guaranteed to have three sets of mutually exclusive indices, that all index into
        # mesh.points:
        # fixed_indices are all indices of vertices that will not be moved,
        # driver_indices are indices of vertices that will get an attached parameter
        # drivens_driver is a dict whose keys are indices of driven vertices and whose value is the index of the
        # driver to control this driven vertex.
        self.driver_indices = np.array(driver_indices)
        self.driven_indices = np.array(list(driven_indices))
        self.fixed_indices = np.array(fixed_indices)

        # set some state variables, so we can avoid doing unnecessary gathers if we don't need to
        self.any_fixed = len(fixed_indices) > 0
        self.any_driven = len(drivens_driver) > 0
        if self.any_driven or self.any_fixed:
            self.full_params = False
        else:
            self.full_params = True

        self.movable_indices = np.setdiff1d(np.arange(mesh.points.shape[0]), fixed_indices)
        movable_count = len(self.movable_indices)
        param_count = len(driver_indices)
        self.fixed_points = mesh.points[fixed_indices]
        self.zero_points = mesh.points[self.movable_indices]

        if self.any_driven:
            # Sanity check that every vertex in movable indices is a key in drivens_driver and that the set of values in
            # drivens_driver is equivalent to the set of driver indices
            assert set(self.movable_indices) == set(drivens_driver.keys()), (
                "ParametricTriangleOptic: movable_indices does not match the domain of the index map."
            )
            assert set(driver_indices) == set(drivens_driver.values()), (
                "ParametricTriangleOptic: driver_indices does not match the set of drivers in the index map."
            )

            # Everything above this line indexes into mesh.points, but to construct the driven-driver gather
            # we need to index relative to drivers and zero_points/movable.
            reindex_driven = {}
            count = 0
            for m in self.movable_indices:
                reindex_driven[m] = count
                count += 1

            reindex_driver = {}
            count = 0
            for d in driver_indices:
                reindex_driver[d] = count
                count += 1

            gather_drivens_driver = {
                reindex_driven[key]: reindex_driver[value] for key, value in drivens_driver.items()
            }

            # Now need to construct the gather_params, a set of indices that can be used to gather from parameters to
            # zero points. (points each zero point to its parameter)
            self.gather_params = tf.constant(
                [gather_drivens_driver[key] for key in range(movable_count)],
                dtype=tf.int32
            )

        # Tile.
        unpacked_faces = mt.unpack_faces(mesh.faces)
        tiling_params = self._compute_tiling(mesh.points, unpacked_faces)
        if tiling_params is not None:
            # indices sets need a particular tiling function, as they need to get tiled and then offset by the
            # total number of vertices.  Uses full mesh tiling params.
            # driver indices is special.  We don't want to add any driver indices - we want these to be driven.
            # So create a set of the tiled driver indices, remove the real driver indices, and add them to the driven
            # indices.
            tiled_driver_indices = set(self._perform_index_tiling(self.driver_indices, tiling_params))
            driven_indices = set(self._perform_index_tiling(self.driven_indices, tiling_params))
            self.driven_indices = np.array((tiled_driver_indices - set(self.driver_indices)) | driven_indices)
            self.fixed_indices = self._perform_index_tiling(self.fixed_indices, tiling_params)
            self.movable_indices = self._perform_index_tiling(self.movable_indices, tiling_params)
            # fixed_points and zero_points have to be tiled like vertices, and unfortunately need their own tiling
            # params.  Can feed an empty tuple for faces, since it won't be needed for either tile.
            self.fixed_points = self._perform_vertex_tiling(
                self.fixed_points,
                self._compute_tiling(self.fixed_points, ())
            )
            self.zero_points = self._perform_vertex_tiling(
                self.zero_points,
                self._compute_tiling(self.zero_points, ())
            )
            # gather_params simply has to be copied / tiled
            self.gather_params = tf.tile(self.gather_params, (tiling_params[0],))
        # raw_faces has to be tiled using the tiling params from the full mesh
        self._raw_faces = self._perform_face_tiling(unpacked_faces, tiling_params)
        self.tiled_base_mesh = pv.PolyData(
            np.concatenate((self.fixed_points, self.zero_points), axis=0),
            mt.pack_faces(self._raw_faces)
        )

        if self.any_fixed:
            # Now need to construct gather_vertices, a set of indices that can be used to gather from
            # the concatenation of fixed_points and zero_points (in that order) to the full vertices.
            gather_vertices = np.argsort(np.concatenate((self.fixed_indices, self.movable_indices)))
            self.gather_vertices = tf.constant(gather_vertices, dtype=tf.int32)

        # Make the final components of the surface.
        self.parameters = tf.Variable(initial_value=tf.zeros((param_count, 1), dtype=tf.float64))
        self.vectors = self.vector_generator(self.zero_points)

        # Need to construct v_to_p, which maps every vertex to a parameter, or -1 if there is no parameter.
        # Want it to be an array, for speed.  Goes like how the gathers are used in get_vertices_from_params, though
        # this constructs a map that is the inverse of the gather maps.
        p = np.arange(param_count, dtype=np.int32)
        if self.any_driven:
            m_to_p = p[self.gather_params.numpy()]
        else:
            m_to_p = p
        if self.any_fixed:
            all_v = np.concatenate((-np.ones((self.fixed_points.shape[0],), dtype=np.int32), m_to_p), axis=0)
            self.v_to_p = all_v[self.gather_vertices.numpy()]
        else:
            self.v_to_p = m_to_p

        # Set the initials
        if initials is None:
            self.initials = tf.zeros((param_count, 1), dtype=tf.float64)
        else:
            try:
                self.initials = tf.broadcast_to(initials, (param_count, 1))
            except Exception:
                initials = np.asarray(initials, dtype=np.float64)
                # If initials cannot be broadcast to the parameters directly, it needs to have one element per
                # vertex, and we will need to match vertices to parameters.  This can be accomplished with v_to_p.
                # v_to_p has one element per vertex in the full mesh.  If the element in a position is positive, it
                # is the index of a parameter, and if the element is -1, it corresponds to a fixed point.

                # Select out the movable points, and relate to parameters.  Harder because, unlike the fixed points,
                # each parameter may be pointed to by more than one vertex.  Will use np.unique and just use the
                # first initial for each parameter, ignoring all the rest of the data.  Unique_p indexes into
                # parameters, but contains -1s for fixed indices that have to be filtered out.  np.unique sorts
                # its output, so it doesn't have to be re-indexed.  unique_p_indices is the index in initials to get
                # for each parameter.
                unique_p, unique_p_indices = np.unique(self.v_to_p, return_index=True)
                unique_p_indices = unique_p_indices[unique_p >= 0]
                self.initials = tf.reshape(tf.convert_to_tensor(initials[unique_p_indices], dtype=tf.float64), (-1, 1))
        self.param_assign(self.initials)

        # Need to temporarily disable the vum so we can update so we can get the parts we need to make the new VUM.
        enable_vum = self.enable_vum
        self.enable_vum = False
        self.update()
        self.enable_vum = enable_vum
        self.try_mesh_tools()
        self.smoother = self.get_smoother(self.settings.smooth_stddev)

    @staticmethod
    def _perform_index_tiling(indices, tiling_params):
        """
        Tile a set of indices with the tiling parameters.  Feed the return from _compute_tiling directly
        to this function.  If there is no tiling to perform, this will return indices as is, so this function can be
        used inline without any case checking.
        """
        if tiling_params is None:
            return indices
        else:
            total_repeats, base_vertex_count, base_face_count, combined_offsets = tiling_params
            # For vertices, need to tile the vertices to the repeats, and add the offsets (which are the repeats)
            # to the vertices.
            return np.tile(indices, total_repeats) + \
                np.repeat(np.arange(total_repeats), len(indices)) * base_vertex_count

    def update_tiling(self):
        self.from_mesh(self.base_mesh, self.parameters)
        self.update_and_redraw()

        #self.v_to_p = np.tile(self.v_to_p, total_repeats)

    def try_mesh_tools(self):
        vertices = self.get_vertices_from_params(tf.zeros_like(self.parameters))
        if self.enable_vum:
            self.vum = tf.constant(mt.cosine_vum(
                self.settings.vum_origin,
                tuple(each.numpy() for each in self.face_vertices),
                self.vertices.numpy(),
                self.faces
            ), dtype=tf.bool)
        if self.enable_accumulator:
            active_vertices = tf.gather(vertices, self.driver_indices, axis=0)

            self.accumulator = mt.cosine_acum(
                tf.constant(self.settings.accumulator_origin, dtype=tf.float64),
                active_vertices
            )

        if self.mt_controller:
            self.mt_controller.update_displays()

    def update(self, force=False):
        if not self.settings.frozen or force:
            # Apply constraints
            for constraint in self.constraints:
                constraint(self)

            # Faces is set in from_mesh.
            # Compute the vertices, from the zero points and parameters
            self.vertices = self.get_vertices_from_params(self.parameters)
            # Separate the vertices out by face
            self.gather_faces()
            # Try applying the vum on the separated vertices
            self.face_vertices = self.try_vum(*self.face_vertices)

            self.compute_norm()

    def get_vertices_from_params(self, params):
        if self.any_driven:
            # If any vertices are driven, we need expand the parameters to match the shape of the zero points.
            gathered_params = tf.gather(params, self.gather_params, axis=0)
        else:
            gathered_params = params

        # Perform transformations
        if self.translation is not None and self.rotation is not None:
            rotated_vectors = tf.convert_to_tensor(self.rotation.rotate(self.vectors), dtype=tf.float64)
            rotated_zero_points = tf.convert_to_tensor(self.rotation.rotate(self.zero_points), dtype=tf.float64)
            vertices = rotated_zero_points + self.translation + gathered_params * rotated_vectors
        else:
            rotated_vectors = tf.convert_to_tensor(self.vectors, dtype=tf.float64)
            vertices = tf.convert_to_tensor(self.zero_points, dtype=tf.float64) + gathered_params * rotated_vectors

        if self.any_fixed:
            # If any vertices are fixed vertices, we need to add them in using gather_vertices
            rotated_fixed_points = tf.convert_to_tensor(self.rotation.rotate(self.fixed_points), dtype=tf.float64)
            all_vertices = tf.concat((rotated_fixed_points + self.translation, vertices), axis=0)
            vertices = tf.gather(all_vertices, self.gather_vertices, axis=0)
        return vertices

    def find_closest_parameter_to(self, point):
        p_indices = tf.range(self.parameters.shape[0])
        if self.any_driven:
            # If any vertices are driven, we need expand the parameters to match the shape of the zero points.
            p_indices = tf.gather(p_indices, self.gather_params, axis=0)
        # p_indices should now be a list of parameter indices that matches up to movable_vertices

        v_index = mt.get_closest_point(self.zero_points, point)
        return p_indices[v_index]

    def constrain(self):
        if not self.settings.frozen:
            for constraint in self.constraints:
                constraint(self)

    def param_assign(self, val):
        self.parameters.assign(val)
        if self._qt_compat:
            self.parameters_updated.sig.emit()

    def param_assign_sub(self, val):
        self.parameters.assign_sub(val)
        if self._qt_compat:
            self.parameters_updated.sig.emit()

    def param_assign_add(self, val):
        self.parameters.assign_add(val)
        if self._qt_compat:
            self.parameters_updated.sig.emit()

    def try_vum(self, first_points, second_points, third_points):
        if not self.enable_vum:
            return first_points, second_points, third_points

        if not self.settings.vum_active or self.vum is None:
            return first_points, second_points, third_points

        first_updatable, second_updatable, third_updatable = \
            tf.unstack(self.vum, axis=1)

        first_updatable = tf.reshape(first_updatable, (-1, 1))
        second_updatable = tf.reshape(second_updatable, (-1, 1))
        third_updatable = tf.reshape(third_updatable, (-1, 1))

        first_points = tf.where(
            first_updatable, first_points, tf.stop_gradient(first_points))
        second_points = tf.where(
            second_updatable, second_points, tf.stop_gradient(second_points))
        third_points = tf.where(
            third_updatable, third_points, tf.stop_gradient(third_points))

        return first_points, second_points, third_points

    def try_accumulate(self, delta):
        if self.enable_accumulator:
            if self.settings.accumulator_active:
                if self.accumulator is not None:
                    return tf.matmul(self.accumulator, delta)
        return delta

    def make_d_mat(self):
        vs = self.get_vertices_from_params(tf.zeros_like(self.parameters))
        avs = tf.gather(vs, self.driver_indices, axis=0)

        self._d_mat = tf.sqrt(tf.reduce_sum((avs[None, :, :] - avs[:, None, :]) ** 2, axis=-1))

    def get_smoother(self, stddev):
        if self._d_mat is None:
            self.make_d_mat()
        elif self._d_mat.shape != (self.parameters.shape[0], self.parameters.shape[0]):
            self.make_d_mat()

        # Gaussian smoother, which is a matrix that can be left-multiplied with parameters
        smoother = tf.exp(-.5 * (self._d_mat / stddev) ** 2)
        # Normalize, so that the average position of the surface is conserved.  I spent a bunch of time fretting about
        # this - couldn't get it to actually normalize the right (second) axis.  But months later I realized... I needed
        # a reshape, so that the norm broadcasts to the correct axis.
        smoother /= tf.reshape(tf.reduce_sum(smoother, axis=1), (-1, 1))
        return smoother

    def smooth(self, smoother=None):
        if smoother is None:
            smoother = self.smoother
        if smoother is not None:
            if self.settings.smooth_active:
                self.param_assign(tf.matmul(smoother, self.parameters))

    def _process_constraints(self, constraints):
        if constraints is None:
            constraints = []
        else:
            constraints = constraints
        for c in constraints:
            if isinstance(c, qtw.QWidget) and self.driver.driver_type == "client":
                self.controller_widgets.append(c)
        return constraints


class ReparameterizableTriangleOpticBase(ParametricTriangleOptic):
    """
    Optic that can change its zero point mesh while preserving its shape.

    This optic is only valid for surfaces that can be collapsed into one of the three coordinate planes (xy, yz, xz) and
    be interpolated from that plane into 3D.  The zero points will live within the chosen coordinate plane and the
    parameters will be built from a vector perpendicular to this plane.

    This class derives from ParametricTriangleOptic and should implement its full interface.  This is class will behave
    identically to a ParametricTriangleOptic until the remesh controls are used to change the underlying base mesh.

    This mesh can adapt its parameters to the new shape of the zero points only while it is running.  It is not possible
    to develop and save a set of parameters, remesh, and then load those parameters, because they will have the wrong
    shape.  Thus the best practice for maintaining compatibility between old solutions and new mesh shapes is to save
    the output mesh rather than the parameters, and load previous output meshes rather than previously saved parameters.
    Of course, loading previous parameter sets does work so long as the shape of the mesh has not changed.
    """
    plane_lookup = {
        "xy": (0, 1, 2, (0.0, 0.0, 1.0)),
        "yz": (1, 2, 0, (1.0, 0.0, 0.0)),
        "xz": (0, 2, 1, (0.0, 1.0, 0.0))
    }

    def __init__(
        self,
        driver,
        system_path,
        settings,
        plane,
        mesh_generator,
        **kwargs
    ):
        """

        Parameters
        ----------

        driver : client.OpticClientWindow
            The top level client window.
        system_path : str
            The path to the parent directory of the optical script.
        settings : dict or settings.Settings
            The settings for this optical element.
        plane : str
            Must be 'xy', 'yz' or 'xz'.  The coordinate plane to use for projection / interpolation.
        mesh_generator : str or MeshGenerator.
            This object controls how new meshes are generated.  It may be a string, in which case a built-in generator
            will be used.  Valid built-in values are:
            'circle': mesh_tools.circular_mesh is used to generate the zero point mesh.
            'square': pyvista.Plane is used to generate the zero point mesh.
            Or it can be a custom MeshGenerator.
        kwargs :
            All remaining kwargs are passed to ParametricTriangleOptic.
        """
        try:
            self.c1, self.c2, self.c3, self.plane_n = self.plane_lookup[plane]
            self.base_plane = plane
        except KeyError as e:
            raise ValueError("ReparameterizablePlanarOptic: plane must be 'xy', 'yz', or 'xz'.") from e
        vector_generator = FromVectorVG(self.plane_n)

        if type(mesh_generator) is str:
            if mesh_generator == "circle":
                self.mesh_generator = CircleMeshGenerator(self)
            elif mesh_generator == "square":
                self.mesh_generator = SquareMeshGenerator(self)
            else:
                raise ValueError(
                    f"ReparameterizablePlanarOptic: {mesh_generator} is not a valid built-in option for mesh_generator."
                )
        elif isinstance(mesh_generator, MeshGenerator):
            self.mesh_generator = mesh_generator
        else:
            raise ValueError(
                "ReparameterizablePlanarOptic: mesh_generator must be a string and valid name for a built-in "
                "generator or an instance of MeshGenerator."
            )

        super().__init__(driver, system_path, settings, vector_generator, **kwargs)
        self.controller_widgets.append(self.mesh_generator.get_ui_widget())

    def remesh(self, new_zero_mesh):
        self.tcp_ack_remeshed = True
        self.p_controller_ack_remeshed = True

        # Remove the transformations.  Save them to re-apply at the end
        saved_translation = self.settings.translation
        self.settings.translation = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.update_translation(False)
        saved_rotation = self.settings.rotation
        self.settings.rotation = TriangleOptic.BASE_ORIENTATION.copy()
        self.update_rotation(False)
        self.update()

        self.remesh_core(new_zero_mesh)

        # Re-apply the transformations
        self.settings.translation = saved_translation
        self.update_translation(False)
        self.settings.rotation = saved_rotation
        self.update_rotation(False)
        self.update()
        self.redraw()

    def remesh_core(self, new_zero_mesh):
        raise NotImplementedError(
            "Cannot use ReparameterizableTriangleOpticBase - it is a base class.  Used a derived class instead."
        )

    def shape_to_p(self, new_zero_mesh):
        raise NotImplementedError(
            "Cannot use ReparameterizableTriangleOpticBase - it is a base class.  Used a derived class instead."
        )


class ReparameterizablePlanarOptic(ReparameterizableTriangleOpticBase):
    def shape_to_p(self):
        """
        Converts a shaped zero point mesh (with extent in the direction of the vectors) to a flat zero point mesh
        with the parameters equal to initial flattened coordinate.  This will not change the shape of the mesh, but
        will make it work better with constraints.
        """
        new_initials = self.vertices[:, self.c3]
        new_zero_mesh = self.base_mesh.copy()

        # Only want to flatten the non-fixed vertices
        if hasattr(self, "filter_fixed"):
            new_zero_mesh.points[:, self.c3][np.logical_not(self.filter_fixed(self.vertices))] = 0
        else:
            new_zero_mesh.points[:, self.c3] = 0

        # Rebuild the optic.
        self.from_mesh(new_zero_mesh, new_initials)

    def remesh_core(self, new_zero_mesh):
        # Project and extract the shape of the surface, and interpolate.
        x, y, z = (self.vertices[:, c].numpy() for c in (self.c1, self.c2, self.c3))
        interpolation = scipy.interpolate.LinearNDInterpolator((x, y), z, 0)
        new_zero_x, new_zero_y = new_zero_mesh.points[:, self.c1], new_zero_mesh.points[:, self.c2]
        new_initials = interpolation((new_zero_x, new_zero_y))

        # New_zero_x, new_zero_y, and new_initials hold what should be the new shape of the mesh.  But new_initials
        # will only be used for driving vertices - the fixed ones will be missed.  Will need to push the fixed points
        # z values (new_initials) into new_zero_mesh now to preserve them.
        if hasattr(self, "filter_fixed"):
            fixed_points = self.filter_fixed(new_zero_mesh.points)
            new_zero_mesh.points[:, self.c3] = np.where(
                fixed_points,
                new_initials,
                0.0
            )

        # Rebuild the optic.
        new_initials = tf.convert_to_tensor(new_initials, dtype=tf.float64)
        self.from_mesh(new_zero_mesh, new_initials)


# ======================================================================================================================


class ParamsUpdatedSignal(qtc.QObject):
    sig = qtc.pyqtSignal()


class TrigBoundaryDisplayController(qtw.QWidget):
    _valid_colors = set(pv.hexcolors.keys())

    def __init__(self, component, plot):
        super().__init__()
        self.component = component
        self._permit_draws = False
        self._params_valid = hasattr(self.component, "parameters")
        self.plot = plot
        self._driving_labels_actors = []

        # define the default display settings for this component
        if self._params_valid:
            drawing_settings = self.component.settings.establish_defaults(
                visible=True,
                norm_arrow_visibility=False,
                norm_arrow_length=0.1,
                parameter_arrow_visibility=False,
                parameter_arrow_length=0.1,
                color="cyan",
                show_edges=False
            )
        else:
            drawing_settings = self.component.settings.establish_defaults(
                visible=True,
                norm_arrow_visibility=False,
                norm_arrow_length=0.1,
                color="cyan",
                show_edges=False
            )

        self.drawer = drawing.TriangleDrawer(plot, component, **self.component.settings.get_subset(drawing_settings))
        self.component.drawer = self.drawer

        # build the UI elements
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(11, 11, 0, 11)
        self.setLayout(main_layout)

        # visibility check box
        self.build_check_box(main_layout, 0, "visible")

        # color line edit
        main_layout.addWidget(qtw.QLabel("color"), 1, 0)
        color_widget = qtw.QLineEdit(self)
        color_widget.setText(str(self.component.settings.color))
        color_widget.editingFinished.connect(self.change_color)
        main_layout.addWidget(color_widget, 1, 1)
        self.color_widget = color_widget

        # opacity entry box
        self.opacity_widget = cw.SettingsEntryBox(
            self.component.settings, "opacity", float, qtg.QDoubleValidator(0, 1, 3), self.change_opacity
        )
        main_layout.addWidget(self.opacity_widget, 2, 0)

        # edges check box
        self.build_check_box(main_layout, 3, "show_edges", self.drawer.rebuild)

        # norm arrows controls
        self.build_check_box(main_layout, 4, "norm_arrow_visibility")
        self.build_entry_box(main_layout, 5, "norm_arrow_length", float, qtg.QDoubleValidator(0, 1e6, 5))

        if self._params_valid:
            self.build_check_box(main_layout, 6, "parameter_arrow_visibility")
            self.build_entry_box(main_layout, 7, "parameter_arrow_length", float, qtg.QDoubleValidator(0, 1e6, 5))
            self.component.settings.establish_defaults(show_driving_labels=False)
            self.build_check_box(main_layout, 8, "show_driving_labels", extra_callback=self.click_driving_labels)

        self._permit_draws = True

    def redraw(self):
        if self._permit_draws:
            self.drawer.draw()

    def change_color(self):
        color = self.color_widget.text()
        if color in self._valid_colors:
            self.component.settings.color = color
            self.drawer.color = color
            self.drawer.rebuild()
            self.color_widget.setStyleSheet("QLineEdit { background-color: white}")
        else:
            self.color_widget.setStyleSheet("QLineEdit { background-color: pink}")

    def change_opacity(self):
        self.drawer.opacity = self.component.settings.opacity
        self.drawer.rebuild()

    def remove_drawer(self):
        self.drawer.delete()
        for each in self._driving_labels_actors:
            self.plot.remove_actor(each)
        self._driving_labels_actors = []

    def build_check_box(self, main_layout, layout_index, name, extra_callback=None):
        main_layout.addWidget(qtw.QLabel(str(name).replace("_", " ")), layout_index, 0)
        widget = qtw.QCheckBox("")

        def callback(state):
            state = bool(state)
            self.component.settings.dict[name] = state
            setattr(self.drawer, name, state)
            if extra_callback is not None:
                extra_callback()
            self.redraw()

        widget.setCheckState(self.component.settings.dict[name])
        callback(self.component.settings.dict[name])
        widget.setTristate(False)
        widget.stateChanged.connect(callback)
        main_layout.addWidget(widget, layout_index, 1)

    def build_entry_box(self, main_layout, layout_index, name, value_type, validator=None):
        main_layout.addWidget(qtw.QLabel(name), layout_index, 0)
        widget = qtw.QLineEdit(self)
        if validator:
            widget.setValidator(validator)

        def callback():
            value = value_type(widget.text())
            self.component.settings.dict[name] = value
            setattr(self.drawer, name, value)
            self.redraw()

        widget.setText(str(self.component.settings.dict[name]))
        callback()
        widget.editingFinished.connect(callback)
        main_layout.addWidget(widget, layout_index, 1)

    def click_driving_labels(self):
        if self.component.settings.show_driving_labels:
            if self.component.driver_indices.shape[0] > 0:
                mesh = pv.PolyData(self.component.vertices.numpy()[self.component.driver_indices])
                actor = self.plot.add_mesh(mesh, render_points_as_spheres=True, color="green", point_size=10.0)
                self._driving_labels_actors.append(actor)
            if self.component.driven_indices.shape[0] > 0:
                mesh = pv.PolyData(self.component.vertices.numpy()[self.component.driven_indices])
                actor = self.plot.add_mesh(mesh, render_points_as_spheres=True, color="yellow", point_size=10.0)
                self._driving_labels_actors.append(actor)
            if self.component.fixed_indices.shape[0] > 0:
                mesh = pv.PolyData(self.component.vertices.numpy()[self.component.fixed_indices])
                actor = self.plot.add_mesh(mesh, render_points_as_spheres=True, color="red", point_size=10.0)
                self._driving_labels_actors.append(actor)
        else:
            for each in self._driving_labels_actors:
                self.plot.remove_actor(each)
            self._driving_labels_actors = []


class OpticController(qtw.QWidget):
    def __init__(self, component, driver, system_path):
        """
        It is the responsibility of the user to establish default settings.

        Always required for a controller driven optic:
            mesh_output_path
        If this is a parametric optic, requires:
            zero_points_input_path
            parameters_path
        If a mesh was not specified to the component constructor:
            mesh_input_path

        Parameters
        ----------
        component : optics.OpticBase
            The optic to associate this controller with
        """
        super().__init__()
        self.component = component
        self.driver = driver
        self._params_valid = hasattr(self.component, "parameters")
        settings_keys = set(self.component.settings.dict.keys())
        self._input_valid = "mesh_input_path" in settings_keys
        self._output_valid = "mesh_output_path" in settings_keys

        # build the UI elements
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(11, 11, 0, 11)
        self.setLayout(main_layout)
        ui_row = 0

        if self._output_valid:
            main_layout.addWidget(qtw.QLabel("Mesh Output"), ui_row, 0, 1, 12)
            ui_row += 1
            main_layout.addWidget(cw.SettingsFileBox(
                self.component.settings, "mesh_output_path", system_path, "*.stl", "save", self.save_mesh
            ), ui_row, 0, 1, 12)
            ui_row += 1
        if self._params_valid:
            self.component.settings.establish_defaults(
                parameters_path=str(pathlib.Path(system_path) / "parameters.dat")
            )
            main_layout.addWidget(qtw.QLabel("Parameters"), ui_row, 0, 1, 12)
            ui_row += 1
            main_layout.addWidget(cw.SettingsFileBox(
                self.component.settings, "parameters_path", system_path, "*.dat", "both",
                self.save_parameters, self.load_parameters
            ), ui_row, 0, 1, 12)
            ui_row += 1
        if self._input_valid:
            main_layout.addWidget(qtw.QLabel("Mesh Input"), ui_row, 0, 1, 12)
            ui_row += 1
            main_layout.addWidget(cw.SettingsFileBox(
                self.component.settings, "mesh_input_path", system_path, "*.stl", "load", self.load_mesh
            ), ui_row, 0, 1, 12)
            ui_row += 1
        if self._output_valid and self._params_valid:
            save_all_button = qtw.QPushButton("Save Everything")
            save_all_button.clicked.connect(self.save_all)
            main_layout.addWidget(save_all_button, ui_row, 0, 1, 3)
        ui_row += 1

        main_layout.addWidget(cw.SettingsCheckBox(self.component.settings, "active", "Active"), ui_row, 0, 1, 3)
        main_layout.addWidget(cw.SettingsCheckBox(self.component.settings, "frozen", "Freeze Optic"), ui_row, 4, 1, 3)
        main_layout.addWidget(cw.SettingsCheckBox(
            self.component.settings, "flip_norm", "Flip Norm", self.component.update_and_redraw
        ), ui_row, 8, 1, 3)
        ui_row += 1

        main_layout.addWidget(
            cw.SettingsVectorBox(
                self.component.settings, "Translation", "translation", self.component.update_translation
            ),
            ui_row, 0, 1, 12
        )
        ui_row += 1
        main_layout.addWidget(
            cw.SettingsVectorBox(
                self.component.settings, "Rotation", "rotation", self.component.update_rotation
            ),
            ui_row, 0, 1, 12
        )
        ui_row += 1

        tiling_layout = qtw.QGridLayout()
        tiling_layout.setContentsMargins(0, 0, 0, 0)
        self.tiling_widget = qtw.QWidget()
        self.tiling_widget.setLayout(tiling_layout)
        main_layout.addWidget(cw.SettingsCheckBox(
            self.component.settings, "tileable", "Tile", self.toggle_tiling_visibility
        ))
        ui_row += 1
        main_layout.addWidget(self.tiling_widget, ui_row, 0, 1, 12)
        self.toggle_tiling_visibility()

        tiling_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "tile_1_low_count", int, qtg.QIntValidator(-500, 500),
            self.component.update_tiling, "Tiling 1 Start Count"
        ), 0, 0, 1, 1)
        tiling_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "tile_1_high_count", int, qtg.QIntValidator(-500, 500),
            self.component.update_tiling, "End"
        ), 0, 1, 1, 1)
        tiling_layout.addWidget(cw.SettingsVectorBox(
            self.component.settings, "Tiling 1 Vector", "tile_1_vector", self.component.update_tiling
        ), 1, 0, 1, 2)

        tiling_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "tile_2_low_count", int, qtg.QIntValidator(-500, 500),
            self.component.update_tiling, "Tiling 2 Start Count"
        ), 2, 0, 1, 1)
        tiling_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "tile_2_high_count", int, qtg.QIntValidator(-500, 500),
            self.component.update_tiling, "End"
        ), 2, 1, 1, 1)
        tiling_layout.addWidget(cw.SettingsVectorBox(
            self.component.settings, "Tiling 2 Vector", "tile_2_vector", self.component.update_tiling
        ), 3, 0, 1, 2)

    def update_component(self):
        self.component.update()
        self.component.redraw()

    def save_mesh(self):
        if self._output_valid:
            self.component.save(self.component.settings.mesh_output_path)
            print(f"saved mesh for {self.component.name}: {self.component.settings.mesh_output_path}")

    def load_zero_points(self):
        if self._input_valid:
            self.component.load(self.component.settings.mesh_input_path)
            self.driver.update_optics()
            self.driver.try_auto_redraw()
            print(f"loaded mesh for {self.component.name}: {self.component.settings.mesh_input_path}")

    def save_parameters(self):
        if self._params_valid:
            try:
                with open(self.component.settings.parameters_path, 'wb') as outFile:
                    pickle.dump((
                        self.component.parameters.numpy()
                    ), outFile)
                print(f"saved parameters for {self.component.name}: {self.component.settings.parameters_path}")
            except Exception:
                print(f"Exception while trying to save parameters")
                print(traceback.format_exc())

    def load_parameters(self):
        if self._params_valid:
            try:
                with open(self.component.settings.parameters_path, 'rb') as inFile:
                    params = pickle.load(inFile)
                    if params.shape == self.component.parameters.shape:
                        self.component.param_assign(params)
                    else:
                        print(f"Error: Cannot load parameters - shape does not match for this optic.")
                        return
                print(f"loaded parameters for {self.component.name}: {self.component.settings.parameters_path}")
                self.driver.update_optics()
                self.driver.redraw()
                if self.driver is not None:
                    self.driver.parameters_pane.refresh_parameters()
            except Exception:
                print(f"Exception while trying to load parameters")
                print(traceback.format_exc())

    def load_mesh(self):
        if self._input_valid:
            self.component.load(self.component.settings.mesh_input_path)
            self.driver.update_optics()
            self.driver.redraw()
            try:
                self.component.drawer.draw()
            except Exception:
                pass
            print(f"loaded mesh for {self.component.name}: {self.component.settings.mesh_input_path}")

    def save_all(self):
        # These functions already check whether their operation is valid, and silently do nothing if it isn't
        self.save_mesh()
        self.save_parameters()

    def toggle_tiling_visibility(self):
        if self.component.settings.tileable:
            self.tiling_widget.show()
        else:
            self.tiling_widget.hide()


class MeshTricksController(qtw.QWidget):
    def __init__(self, component, client):
        """
        Controller widget that controls the display and generation of the mesh tricks.  This widget establishes
        its needed defaults.

        Parameters
        ----------
        component : optics.ParametricTriangleOptic
            The component to control with this controller
        """
        super().__init__()
        self.component = component
        self.client = client

        # Build the UI elements
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(11, 11, 0, 11)
        self.setLayout(main_layout)

        # Set up the vum origin
        if self.component.enable_vum:
            self.component.settings.establish_defaults(vum_origin=[0.0, 0.0, 0.0])
            self.vum_origin_mesh = pv.PolyData(np.array(self.component.settings.vum_origin))
            self.vum_origin_actor = self.client.plot.add_mesh(
                self.vum_origin_mesh, render_points_as_spheres=True, color="red", point_size=15.0
            )

            main_layout.addWidget(
                cw.SettingsVectorBox(
                    self.component.settings,
                    "VUM Origin",
                    "vum_origin",
                    [self.update_vum_origin_display, self.component.try_mesh_tools]
                ), 0, 0, 1, 2
            )
            self.component.settings.establish_defaults(vum_origin_visible=False)
            main_layout.addWidget(
                cw.SettingsCheckBox(
                    self.component.settings, "vum_origin_visible", "Show vum origin", self.toggle_vum_origin_display
                ), 2, 1, 1, 1
            )
            self.toggle_vum_origin_display(self.component.settings.vum_origin_visible)

        # Set up the acum origin
        if self.component.enable_accumulator:
            self.component.settings.establish_defaults(accumulator_origin=[0.0, 0.0, 0.0])
            self.acum_origin_mesh = pv.PolyData(np.array(self.component.settings.accumulator_origin))
            self.acum_origin_actor = self.client.plot.add_mesh(
                self.acum_origin_mesh, render_points_as_spheres=True, color="red", point_size=15.0
            )

            main_layout.addWidget(
                cw.SettingsVectorBox(
                    self.component.settings,
                    "Acum Origin",
                    "accumulator_origin",
                    [self.update_acum_origin_display, self.component.try_mesh_tools]
                ), 1, 0, 1, 2
            )
            self.component.settings.establish_defaults(acum_origin_visible=False)
            main_layout.addWidget(
                cw.SettingsCheckBox(
                    self.component.settings, "acum_origin_visible", "Show accumulator origin", self.toggle_acum_origin_display
                ), 2, 0, 1, 1
            )
            self.toggle_acum_origin_display(self.component.settings.acum_origin_visible)

        # Set up the vertex update map
        if self.component.enable_vum:
            self.component.settings.establish_defaults(vum_active=True)
            self.component.settings.establish_defaults(vum_visible=False)
            main_layout.addWidget(
                cw.SettingsCheckBox(self.component.settings, "vum_active", "VUM Active"),
                3, 1, 1, 1
            )
            main_layout.addWidget(
                cw.SettingsCheckBox(self.component.settings, "vum_visible", "VUM Visible", self.update_vum_display),
                4, 1, 1, 1
            )
        else:
            self.component.settings.establish_defaults(vum_active=False)
            self.component.settings.establish_defaults(vum_visible=False)
        self.vum_actor = None

        # Set up the accumulator
        if self.component.enable_accumulator:
            self.component.settings.establish_defaults(accumulator_active=True)
            main_layout.addWidget(
                cw.SettingsCheckBox(self.component.settings, "accumulator_active", "Accumulator Active"),
                3, 0, 1, 1
            )
        else:
            self.component.settings.establish_defaults(accumulator_active=False)

    def toggle_vum_origin_display(self, active):
        self.vum_origin_actor.SetVisibility(bool(active))

    def toggle_acum_origin_display(self, active):
        self.acum_origin_actor.SetVisibility(bool(active))

    def update_vum_origin_display(self):
        self.vum_origin_mesh.points = np.array(self.component.settings.vum_origin)

    def update_acum_origin_display(self):
        self.acum_origin_mesh.points = np.array(self.component.settings.accumulator_origin)

    def update_vum_display(self, active):
        if self.vum_actor is not None:
            self.client.plot.remove_actor(self.vum_actor)
        if active:
            start_points = []
            directions = []
            vertices = self.component.vertices.numpy()

            for face, vu in zip(self.component.faces, self.component.vum):
                for v, on in zip(face, vu):
                    if on:
                        face_center = np.sum(vertices[face], axis=0) / 3
                        direction = vertices[v] - face_center
                        start_points.append(face_center)
                        directions.append(direction)

            start_points = np.array(start_points)
            directions = np.array(directions)

            mesh = pv.PolyData(start_points)
            mesh["vectors"] = directions
            mesh.set_active_vectors("vectors")
            self.vum_actor = self.client.plot.add_mesh(
                mesh.arrows,
                color="yellow",
                reset_camera=False
            )

    def update_displays(self):
        self.update_vum_display(self.component.settings.vum_visible)

    def remove_drawer(self):
        self.client.plot.remove_actor(self.vum_actor)
        self.client.plot.remove_actor(self.vum_origin_actor)
        self.client.plot.remove_actor(self.acum_origin_actor)


# ======================================================================================================================


class ClipConstraint(qtw.QWidget):
    def __init__(self, settings, min, max):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(clip_constraint_min=min, clip_constraint_max=max)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(cw.SettingsRangeBox(
            self.settings, "Clip Constraint Range", "clip_constraint_min", "clip_constraint_max", float,
            validator=qtg.QDoubleValidator(-1e6, 1e6, 9)
        ))
        self.setLayout(layout)

    def __call__(self, component):
        component.param_assign(tf.clip_by_value(
            component.parameters,
            self.settings.clip_constraint_min,
            self.settings.clip_constraint_max
        ))


class ThicknessConstraint(qtw.QWidget):
    def __init__(self, settings, val, min_mode=True):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(thickness_constraint=val, thickness_mode_min=min_mode)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "thickness_constraint", float, validator=qtg.QDoubleValidator(-1e6, 1e6, 9),
            label="Thickness"
        ))
        self.mode_checkbox = cw.SettingsCheckBox(
            self.settings, "thickness_mode_min", "...", callback=self.mode_toggled
        )
        self.mode_toggled()
        layout.addWidget(self.mode_checkbox)
        self.setLayout(layout)

    def mode_toggled(self):
        if self.settings.thickness_mode_min:
            self.mode_checkbox.label.setText("Minimum")
        else:
            self.mode_checkbox.label.setText("Maximum")

    def __call__(self, component):
        if self.settings.thickness_mode_min:
            diff = self.settings.thickness_constraint - tf.reduce_min(component.parameters)
        else:
            diff = self.settings.thickness_constraint - tf.reduce_max(component.parameters)
        component.param_assign_add(tf.broadcast_to(diff, component.parameters.shape))


class PointConstraint(qtw.QWidget):
    """
    This hasn't been rigorously tested with fixed and driven vertices!!!

    Fixed_constraint is a distance in parameter space - it is relative to the zero points
    """
    def __init__(self, settings, fixed_constraint, fixed_center):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(
            fixed_constraint=fixed_constraint, fixed_center=np.array(fixed_center, dtype=np.float64)
        )
        layout = qtw.QVBoxLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "fixed_constraint", float, validator=qtg.QDoubleValidator(-1e6, 1e6, 9),
            label="Fixed Pos"
        ))
        self.mode_checkbox = cw.SettingsVectorBox(
            self.settings, "Fixed Point", "fixed_center", callback=self.moved_fixed
        )
        self._fixed_vertex = None
        self.moved_fixed()
        layout.addWidget(self.mode_checkbox)
        self.setLayout(layout)

    def moved_fixed(self):
        self._fixed_vertex = None

    def __call__(self, component):
        if self._fixed_vertex is None:
            self._fixed_vertex = component.find_closest_parameter_to(component.settings.fixed_center)
        diff = self.settings.fixed_constraint - component.parameters[self._fixed_vertex]
        component.param_assign_add(tf.broadcast_to(diff, component.parameters.shape))


class SpacingConstraint(qtw.QWidget):
    def __init__(self, settings, distance_constraint, target=None, mode="min"):
        super().__init__()
        self.settings = settings
        self.target = target
        if mode not in {"min", "max", "fixed_min", "fixed_max"}:
            raise ValueError(f"FixedMinDistanceConstraint: mode must be 'min' or 'max'. 'fixed_min' or 'fixed_max'.")
        self._mode = mode
        self.settings.establish_defaults(distance_constraint=distance_constraint)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "distance_constraint", float, validator=qtg.QDoubleValidator(-1e6, 1e6, 9)
        ))
        self.setLayout(layout)

    def __call__(self, component):
        if self.target is None:
            target = tf.zeros_like(component.parameters)
        else:
            target = self.target.parameters
        if self._mode == "min":
            diff = tf.where(
                component.parameters - target < self.settings.distance_constraint,
                component.parameters - target - self.settings.distance_constraint,
                tf.zeros_like(target)
            )
        elif self._mode == "max":
            diff = tf.where(
                component.parameters - target > self.settings.distance_constraint,
                component.parameters - target - self.settings.distance_constraint,
                tf.zeros_like(target)
            )
        elif self._mode == "fixed_min":
            diff = tf.reduce_min(component.parameters - target - self.settings.distance_constraint)
        else: # self._mode == "fixed_max
            diff = tf.reduce_max(component.parameters - target - self.settings.distance_constraint)

        diff = tf.broadcast_to(diff, component.parameters.shape)
        component.param_assign_sub(diff)


class StripConstraint(qtw.QWidget):
    _c_LUT = {"x": 0, "y": 1, "z": 2}

    def __init__(self, settings, strip_pos=0.0, strip_width=1.0, strip_height=.5, strip_plane="x-z", min_mode=True):
        super().__init__()
        self.settings = settings
        self._cw = 0
        self._ch = 1
        self.settings.establish_defaults(
            strip_pos=strip_pos, strip_width=strip_width, strip_height=strip_height, strip_mode=min_mode,
            strip_plane=strip_plane
        )
        layout = qtw.QGridLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(qtw.QLabel("Strip Constraint"), 0, 0, 1, 2)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "strip_pos", float, label="Strip Position",
            validator=qtg.QDoubleValidator(-1e3, 1e3, 6)
        ), 1, 0, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "strip_width", float, label="Strip Width",
            validator=qtg.QDoubleValidator(1e-3, 1e3, 6)
        ), 1, 1, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "strip_height", float, label="Strip Height",
            validator=qtg.QDoubleValidator(1e-3, 1e3, 6)
        ), 2, 0, 1, 1)
        layout.addWidget(cw.SettingsComboBox(
            self.settings, "Width-Height", "strip_plane", ("x-y", "x-z", "y-x", "y-z", "z-x", "z-y"),
            callback=self._update_plane
        ), 2, 1, 1, 1)
        layout.addWidget(cw.SettingsCheckBox(self.settings, "strip_mode", "Minimum"), 3, 0, 1, 1)
        self.setLayout(layout)

    def _update_plane(self):
        self._cw = self._c_LUT[self.settings.strip_plane[0]]
        self._ch = self._c_LUT[self.settings.strip_plane[2]]

    def __call__(self, component):
        # Have to do this because vertices gets set after constraints are processed, so have to look forward to
        # get the vertices.
        vertices = component.get_vertices_from_params(component.parameters)
        x = vertices[:, self._cw]
        y = vertices[:, self._ch]
        in_strip = tf.abs(x - component.settings.strip_pos) < component.settings.strip_width / 2.0
        if component.settings.strip_mode:
            # minimum mode
            delta = component.settings.strip_height - y
            delta = tf.where(tf.logical_and(delta > 0.0, in_strip), delta, 0.0)
            extrema = tf.reduce_max
        else:
            # maximum mode
            delta = component.settings.strip_height - y
            delta = tf.where(tf.logical_and(delta < 0.0, in_strip), delta, 0.0)
            extrema = tf.reduce_min

        # delta is where each vertex needs to change, but we need to convert this to parameters, which can be done
        # with v_to_p.
        vertex_param, parameter_index = tf.meshgrid(component.v_to_p, tf.range(component.parameters.shape[0]))
        mask = vertex_param == parameter_index

        delta_p = tf.zeros(component.parameters.shape, dtype=tf.float64)
        delta, delta_p = tf.meshgrid(delta, delta_p)

        delta_p = tf.where(mask, delta, delta_p)
        delta_p = extrema(delta_p, axis=1, keepdims=True)

        component.param_assign_add(delta_p)


class SpotConstraint(qtw.QWidget):
    def __init__(self, settings, center=(0.0, 0.0), height=0.5, radius=.1, plane="xz", min_mode=True):
        super().__init__()
        self.settings = settings
        self._c1 = 0
        self._c2 = 2
        self._ch = 1
        self.settings.establish_defaults(
            spot_center_1=center[0], spot_center_2=center[1], spot_height=height, spot_radius=radius, spot_plane=plane,
            spot_mode=min_mode,
        )
        layout = qtw.QGridLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        layout.addWidget(qtw.QLabel("Spot Constraint"), 0, 0, 1, 2)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "spot_center_1", float, label="Center C1",
            validator=qtg.QDoubleValidator(-1e3, 1e3, 6)
        ), 1, 0, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "spot_center_2", float, label="Center C2",
            validator=qtg.QDoubleValidator(1e-3, 1e3, 6)
        ), 1, 1, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "spot_height", float, label="Spot Height",
            validator=qtg.QDoubleValidator(1e-3, 1e3, 6)
        ), 2, 0, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "spot_radius", float, label="Spot Radius",
            validator=qtg.QDoubleValidator(1e-3, 1e3, 6)
        ), 2, 1, 1, 1)
        layout.addWidget(cw.SettingsComboBox(
            self.settings, "Plane", "spot_plane", ("xy", "yz", "xz"),
            callback=self._update_plane
        ), 3, 0, 1, 1)
        layout.addWidget(cw.SettingsCheckBox(self.settings, "spot_mode", "Minimum"), 3, 1, 1, 1)
        self.setLayout(layout)

    def _update_plane(self):
        if self.settings.spot_plane == "xy":
            self._c1 = 0
            self._c2 = 1
            self._ch = 2
        elif self.settings.spot_plane == "yz":
            self._c1 = 1
            self._c2 = 2
            self._ch = 0
        else: #self.settings.spot_plane == "xz":
            self._c1 = 0
            self._c2 = 2
            self._ch = 1

    def __call__(self, component):
        # Have to do this because vertices gets set after constraints are processed, so have to look forward to
        # get the vertices.
        vertices = component.get_vertices_from_params(component.parameters)
        x = vertices[:, self._c1] - self.settings.spot_center_1
        y = vertices[:, self._c2] - self.settings.spot_center_2
        z = vertices[:, self._ch]
        in_spot = x*x + y*y < self.settings.spot_radius*self.settings.spot_radius
        if component.settings.strip_mode:
            # minimum mode
            delta = component.settings.spot_height - z
            delta = tf.where(tf.logical_and(delta > 0.0, in_spot), delta, 0.0)
            extrema = tf.reduce_max
        else:
            # maximum mode
            delta = component.settings.spot_height - z
            delta = tf.where(tf.logical_and(delta < 0.0, in_spot), delta, 0.0)
            extrema = tf.reduce_min

        # delta is where each vertex needs to change, but we need to convert this to parameters, which can be done
        # with v_to_p.
        vertex_param, parameter_index = tf.meshgrid(component.v_to_p, tf.range(component.parameters.shape[0]))
        mask = vertex_param == parameter_index

        delta_p = tf.zeros(component.parameters.shape, dtype=tf.float64)
        delta, delta_p = tf.meshgrid(delta, delta_p)

        delta_p = tf.where(mask, delta, delta_p)
        delta_p = extrema(delta_p, axis=1, keepdims=True)

        component.param_assign_add(delta_p)


# ======================================================================================================================


class MeshGenerator(abc.ABC):
    def __init__(self, optic, max_ui_cols=12):
        self.optic = optic
        self.max_ui_cols = max_ui_cols
        self.remesh_button = qtw.QPushButton("Remesh")
        self.extract_button = qtw.QPushButton("Extract Parameters")

    def get_ui_widget(self):
        widget = qtw.QWidget()
        layout = qtw.QGridLayout()
        layout.setContentsMargins(11, 11, 0, 11)
        widget.setLayout(layout)

        ui_row = 0
        layout.addWidget(self.remesh_button, ui_row, 0, 1, int(self.max_ui_cols/2))
        self.remesh_button.clicked.connect(self._remesh)
        layout.addWidget(self.extract_button, ui_row, int(self.max_ui_cols / 2), 1, int(self.max_ui_cols/2))
        self.extract_button.clicked.connect(self._extract_params)
        ui_row += 1

        self.make_ui_widget(layout, ui_row)
        return widget

    @abc.abstractmethod
    def make_ui_widget(self, layout, ui_row):
        """
        Define the remeshing parameters and populate a qt layout with controls that interface with them.

        Parameters
        ----------
        layout : qtw.QGridLayout
            The qtw.QGridLayout that makes up the base widget.  Controls are placed into this layout.
        ui_row : int
            The row at which to start placing widgets.  The remesh button is placed on row 0, and I have adapted
            this practice of keeping track of the row in a variable to make it easier to develop grid UIs.

        """
        return NotImplementedError

    @abc.abstractmethod
    def get_new_mesh(self, optic):
        """
        Generate (and return) a new pyvista mesh to use as the new zero points for the optic.
        """
        return NotImplementedError

    def _remesh(self):
        self.optic.remesh(self.get_new_mesh(self.optic))

    def _extract_params(self):
        self.optic.shape_to_p()


class CircleMeshGenerator(MeshGenerator):
    def make_ui_widget(self, layout, ui_row):
        self.optic.settings.establish_defaults(
            remesh_radius=1.0,
            remesh_target_edge_size=.1
        )

        layout.addWidget(cw.SettingsEntryBox(
            self.optic.settings,
            "remesh_radius",
            float,
            qtg.QDoubleValidator(1e-6, 1e6, 8),
        ), ui_row, 0, 1, 6)
        layout.addWidget(cw.SettingsEntryBox(
            self.optic.settings,
            "remesh_target_edge_size",
            float,
            qtg.QDoubleValidator(1e-6, 1e6, 8),
        ), ui_row, 6, 1, 6)

    def get_new_mesh(self, optic):
        return mt.circular_mesh(
            self.optic.settings.remesh_radius,
            self.optic.settings.remesh_target_edge_size
        )


class SquareMeshGenerator(MeshGenerator):
    def make_ui_widget(self, layout, ui_row):
        self.optic.settings.establish_defaults(
            remesh_i_resolution=5,
            remesh_j_resolution=5
        )

        layout.addWidget(cw.SettingsEntryBox(
            self.optic.settings,
            "remesh_i_resolution",
            int,
            qtg.QIntValidator(1, int(1e6)),
        ), ui_row, 0, 1, 6)
        layout.addWidget(cw.SettingsEntryBox(
            self.optic.settings,
            "remesh_j_resolution",
            int,
            qtg.QIntValidator(1, int(1e6)),
        ), ui_row, 6, 1, 6)

    def get_new_mesh(self, optic):
        # Get the size of the plane to create from the optic
        points = np.array(optic.base_mesh.points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        x_min, x_max = np.amin(x), np.amax(x)
        y_min, y_max = np.amin(y), np.amax(y)
        z_min, z_max = np.amin(z), np.amax(z)
        if optic.base_plane == "xy":
            center = ((x_max + x_min)/2, (y_max + y_min)/2, 0)
            i_size = x_max - x_min
            j_size = y_max - y_min
        elif optic.base_plane == "yz":
            center = (0, (y_min + y_min) / 2, (z_max + z_min) / 2)
            i_size = y_max - y_min
            j_size = z_max - z_min
        else:  # optic.base_plane == "xz"
            center = ((x_max + x_min) / 2, 0, (z_max + z_min) / 2)
            i_size = x_max - x_min
            j_size = z_max - z_min

        return pv.Plane(
            center=center,
            direction=optic.plane_n,
            i_size=i_size,
            j_size=j_size,
            i_resolution=self.optic.settings.remesh_i_resolution,
            j_resolution=self.optic.settings.remesh_j_resolution
        ).triangulate()
