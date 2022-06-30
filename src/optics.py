import tensorflow as tf
import pyvista as pv
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw

import tfrt2.src.mesh_tools as mt
import tfrt2.src.component_widgets as cw


class TriangleOptic:
    """
    Basic 3D optical element composed of a triangulated mesh.

    Please note that vertices and faces are always assumed to be tensorflow tensors.
    """
    def __init__(self, name, client, system_path, settings, mesh=None, mat_in=None, mat_out=None, hold_load=False):
        """
        Basic 3D Triangulated optical surface.

        This class is suitable for non-parametric optics and technical surfaces, like stops or targets.

        It is the user's responsibility to feed and initialize the settings given to this object.  Omitting certain
        settings will suppress behavior in the component controller (can omit displaying IO for a technical surface,
        for instance).  Defaults will only be established by this class if explicitly stated in this documentation
        (though please note that controllers can also establish defaults)

        Parameters
        ----------
        name : str
            A unique name used to identify this component.  Uniqueness will be enforced when this component is
            added to a system.
        client : client.OpticClientWindow
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

        Public Attributes
        -----------------
        Frozen : bool
            Can be set after constructor to prevent future calls to update from doing anything.  This can speed up
            performace for optics that do not change after initialization.

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
        self.settings = settings or settings.Settings()
        self.frozen = False
        self.name = str(name)
        self.mat_in = mat_in
        self.mat_out = mat_out
        self.controller_widgets = []
        self.vertices = None
        self.faces = None
        self.client = client
        self._fields = {}  # COMPAT

        if not hold_load:
            if mesh is None:
                self.load(self.settings.mesh_input_path)
            else:
                self.from_mesh(mesh)

        self.controller_widgets.append(cw.OpticController(self, client, system_path))

    def load(self, in_path):
        self.from_mesh(pv.read(in_path))

    def save(self, out_path):
        self.as_mesh().save(out_path)

    def as_mesh(self):
        return pv.PolyData(self.vertices.numpy(), mt.pack_faces(self.faces.numpy()))

    def from_mesh(self, mesh):
        self.vertices = tf.constant(mesh.points, dtype=tf.float64)
        self.faces = tf.constant(mt.unpack_faces(mesh.faces), dtype=tf.int32)

    def update(self, force=False):
        if not self.frozen or force:
            self.update_fields_from_points(*self.get_points_from_vertices())

    def get_points_from_vertices(self):
        return mt.points_from_vertices(self.vertices, self.faces)

    def update_fields_from_points(self, first_points, second_points, third_points):
        """
        Updates the fields from points.  COMPAT
        """
        self["xp"], self["yp"], self["zp"] = tf.unstack(first_points, axis=1)
        self["x1"], self["y1"], self["z1"] = tf.unstack(second_points, axis=1)
        self["x2"], self["y2"], self["z2"] = tf.unstack(third_points, axis=1)
        self["norm"] = tf.linalg.normalize(
            tf.linalg.cross(
                second_points - first_points,
                third_points - second_points
            ),
            axis=1)[0]
        if self.mat_in is not None:
            self["mat_in"] = tf.broadcast_to(self.mat_in, shape=self["xp"].shape)
        if self.mat_out is not None:
            self["mat_out"] = tf.broadcast_to(self.mat_out, shape=self["xp"].shape)

    def __getitem__(self, key):
        # COMPAT
        return self._fields[key]

    def __setitem__(self, key, value):
        # COMPAT
        self._fields[key] = value

    def keys(self):
        # COMPAT
        return self._fields.keys()

    @property
    def dimension(self):
        return 3


class ParametricTriangleOptic(TriangleOptic):
    """
    3D Triangulated parametric optic, with advanced features for use with gradient descent optimization.
    """
    def __init__(
        self,
        name,
        vector_generator,
        client,
        system_path,
        settings,
        mesh=None,
        mat_in=None,
        mat_out=None,
        enable_vum=True,
        enable_accumulator=True,
        filter_fixed=None,
        filter_drivers=None,
        attach_to_driver=None,
        qt_compat=False
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
        name : str
            A unique name used to identify this component.  Uniqueness will be enforced when this component is
            added to a system.
        vector_generator : callable
            A function that will generate a set of n vectors when given a set of n 3D points.  Used to attach
            vectors to each vertex so they can be moved by the parameters.
        client : client.OpticClientWindow
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
        Frozen : bool
            Can be set after constructor to prevent future calls to update from doing anything.  This can speed up
            performace for optics that do not change after initialization.

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
        self.constraints = []
        self.fixed_points = None
        self.gather_vertices = None
        self.gather_params = None
        self.initials = None
        self.any_fixed = False
        self.any_driven = False
        self.driver_indices = None
        self.full_params = False
        self._qt_compat = qt_compat
        if qt_compat:
            self.parameters_updated = ParamsUpdatedSignal()
        else:
            self.parameters_updated = None

        super().__init__(name, client, system_path, settings, None, mat_in=mat_in, mat_out=mat_out, hold_load=True)

        if self.enable_vum or self.enable_accumulator:
            self.mt_controller = cw.MeshTricksController(self, client)
            self.controller_widgets.append(self.mt_controller)
        else:
            self.mt_controller = None

        # Having some difficulty with initialization order, so need to hold off on loading until now
        if mesh is None:
            self.load(self.settings.mesh_input_path)
        else:
            self.from_mesh(mesh)

    def from_mesh(self, mesh):
        all_indices = np.arange(mesh.points.shape[0])
        available_indices = all_indices.copy()
        drivens_driver = {}

        if hasattr(self, "filter_fixed"):
            fixed_mask = self.filter_fixed(mesh.points)
            fixed_indices = available_indices[fixed_mask]
            available_indices = available_indices[np.logical_not(fixed_mask)]
        else:
            fixed_indices = []

        if hasattr(self, "filter_drivers"):
            driver_mask = self.filter_drivers(mesh.points[available_indices])
            driver_indices = available_indices[driver_mask]
            available_indices = available_indices[np.logical_not(driver_mask)]

            if hasattr(self, "attach_to_driver"):
                available_indices = set(available_indices)
                for driver in driver_indices:
                    attached_indices = all_indices[self.attach_to_driver(mesh.points[driver], mesh.points)]
                    attached_indices = set(attached_indices) & available_indices
                    available_indices -= attached_indices
                    for each in attached_indices:
                        drivens_driver[each] = driver

            # any indices still in available_indices are left over, and need to be added to driver_indices
            driver_indices = np.concatenate((driver_indices, np.array(list(available_indices), dtype=np.int32)))

            # Add each driver as a driver to itself into drivens_driver.
            for driver in driver_indices:
                drivens_driver[driver] = driver
        else:
            driver_indices = available_indices

        # At this point we should be guarenteed to have three sets of mutually exclusive indices:
        # fixed_indices are all indices of vertices that will not be moved,
        # driver_indices are indices of vertices that will get an attached parameter
        # drivens_driver is a dict whose keys are indices of driven vertices and whose value is the index of the
        # driver to controll this driven vertex.

        # set some state variables, so we can avoid doing unnecessary gathers if we don't need to
        self.any_fixed = len(fixed_indices) > 0
        self.any_driven = len(drivens_driver) > 0
        if self.any_driven or self.any_fixed:
            self.full_params = False
        else:
            self.full_params = True
        self.driver_indices = driver_indices

        movable_indices = np.setdiff1d(np.arange(mesh.points.shape[0]), fixed_indices)
        movable_count = len(movable_indices)
        param_count = len(driver_indices)
        self.fixed_points = tf.constant(mesh.points[fixed_indices], dtype=tf.float64)
        self.zero_points = tf.constant(mesh.points[movable_indices], dtype=tf.float64)

        if self.any_driven:
            # Sanity check that every vertex in movable indices is a key in drivens_driver and that the set of values in
            # drivens_driver is equivalent to the set of driver indices
            assert set(movable_indices) == set(drivens_driver.keys()), (
                "ParametricTriangleOptic: movable_indices does not match the domain of the index map."
            )
            assert set(driver_indices) == set(drivens_driver.values()), (
                "ParametricTriangleOptic: driver_indices does not match the set of drivers in the index map."
            )

            # Everything above this line indexes into mesh.points, but to construct the driven-driver gather
            # we need to index relative to drivers and zero_points/movable.
            reindex_driven = {}
            count = 0
            for m in movable_indices:
                reindex_driven[m] = count
                count += 1

            reindex_driver = {}
            count = 0
            for d in driver_indices:
                reindex_driver[d] = count
                count += 1

            drivens_driver = {reindex_driven[key]: reindex_driver[value] for key, value in drivens_driver.items()}

            # Now need to construct the gather_params, a set of indices that can be used to gather from parameters to
            # zero points.
            self.gather_params = tf.constant([drivens_driver[key] for key in range(movable_count)], dtype=tf.int32)

        if self.any_fixed:
            # Now need to construct gather_vertices, a set of indices that can be used to gather from
            # the concatenation of fixed_points and zero_points (in that order) to the indices.
            gather_vertices = np.argsort(np.concatenate((fixed_indices, movable_indices)))
            self.gather_vertices = tf.constant(gather_vertices, dtype=tf.int32)

        # Make the final components of the surface.
        self.initials = tf.zeros((param_count, 1), dtype=tf.float64)
        self.parameters = tf.Variable(self.initials)
        self.faces = tf.constant(mt.unpack_faces(mesh.faces), dtype=tf.int32)
        self.vectors = tf.constant(self.vector_generator(self.zero_points), dtype=tf.float64)

        self.update()
        self.try_mesh_tools()

    def try_mesh_tools(self):
        if self.enable_vum:
            self.vum = mt.cosine_vum(
                tf.constant(self.settings.vum_origin, dtype=tf.float64),
                self.vertices,
                self.faces
            )
        if self.enable_accumulator:
            active_vertices = tf.gather(self.vertices, self.driver_indices, axis=0)
            self.accumulator = mt.cosine_acum(
                tf.constant(self.settings.accumulator_origin, dtype=tf.float64),
                active_vertices
            )

        self.mt_controller.update_displays()

    def update(self, force=False):
        if not self.frozen or force:
            for constraint in self.constraints:
                constraint(self)

            if self.any_driven:
                # If any vertices are driven, we need expand the parameters to match the shape of the zero points.
                gathered_params = tf.gather(self.parameters, self.gather_params, axis=0)
            else:
                gathered_params = self.parameters
            self.vertices = self.zero_points + gathered_params * self.vectors

            if self.any_fixed:
                # If any vertices are fixed vertices, we need to add them in using gather_vertices
                all_vertices = tf.concat((self.fixed_points, self.vertices), axis=0)
                self.vertices = tf.gather(all_vertices, self.gather_vertices, axis=0)

            first, second, third = self.process_vum(*self.get_points_from_vertices())
            self.update_fields_from_points(first, second, third)  # COMPAT

    def constrain(self):
        if not self.frozen:
            for constraint in self.constraints:
                constraint(self)

    def param_assign(self, val):
        try:
            self.parameters.assign(val)
            if self._qt_compat:
                self.parameters_updated.sig.emit()
        except ValueError:
            print(
                "ParametricTriangleOptic: Tried to assign params of the wrong shape.  This might be possible with a "
                "reparametrizable optic instead."
            )

    def param_assign_sub(self, val):
        try:
            self.parameters.assign_sub(val)
            if self._qt_compat:
                self.parameters_updated.sig.emit()
        except ValueError:
            print(
                "ParametricTriangleOptic: Tried to assign_sub params of the wrong shape.  This might be possible "
                "with a reparametrizable optic instead."
            )

    def param_assign_add(self, val):
        try:
            self.parameters.assign_add(val)
            if self._qt_compat:
                self.parameters_updated.sig.emit()
        except ValueError:
            print(
                "ParametricTriangleOptic: Tried to assign_add params of the wrong shape.  This might be possible "
                "with a reparametrizable optic instead."
            )

    def process_vum(self, first_points, second_points, third_points):
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


class ParamsUpdatedSignal(qtc.QObject):
    sig = qtc.pyqtSignal()

