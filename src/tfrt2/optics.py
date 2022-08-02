import tensorflow as tf
import pyvista as pv
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw

import tfrt2.mesh_tools as mt
import tfrt2.component_widgets as cw
import tfrt2.drawing as drawing


class TriangleOptic:
    """
    Basic 3D optical element composed of a triangulated mesh.

    Please note that vertices and faces are always assumed to be tensorflow tensors.
    """
    dimension = 3

    def __init__(
        self, driver, system_path, settings, mesh=None, mat_in=None, mat_out=None, hold_load=False, flip_norm=False
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
        self.name = None
        self.mat_in = mat_in or 0
        self.mat_out = mat_out or 0
        self.vertices = None
        self.faces = None
        self.driver = driver
        self.face_vertices = None
        self.norm = None
        self.flip_norm = flip_norm

        if not hold_load:
            if mesh is None:
                self.load(self.settings.mesh_input_path)
            else:
                self.from_mesh(mesh)

        self.controller_widgets = []
        if self.driver.driver_type == "client":
            self.controller_widgets.append(cw.OpticController(self, driver, system_path))

    def load(self, in_path):
        self.from_mesh(pv.read(in_path))

    def save(self, out_path):
        self.as_mesh().save(out_path)

    def as_mesh(self):
        try:
            return pv.PolyData(self.vertices.numpy(), mt.pack_faces(self.faces.numpy()))
        except Exception:
            return pv.PolyData(np.array(self.vertices), mt.pack_faces(np.array(self.faces)))

    def from_mesh(self, mesh):
        self.vertices = tf.constant(mesh.points, dtype=tf.float64)
        self.faces = tf.constant(self.try_flip_norm(mt.unpack_faces(mesh.faces)), dtype=tf.int32)
        # Separate the vertices out by face
        self.gather_faces()
        # Compute the norm
        self.compute_norm()

    def try_flip_norm(self, faces):
        if self.flip_norm:
            return np.take(faces, [2, 1, 0], axis=1)
        else:
            return faces

    def gather_faces(self):
        self.face_vertices = mt.points_from_faces(self.vertices, self.faces)

    def compute_norm(self):
        first_points, second_points, third_points = self.face_vertices
        self.norm = tf.linalg.normalize(
            tf.linalg.cross(
                second_points - first_points,
                third_points - first_points
            ),
            axis=1)[0]

    def update(self):
        pass


class ParametricTriangleOptic(TriangleOptic):
    """
    3D Triangulated parametric optic, with advanced features for use with gradient descent optimization.

    Update adds some extra steps beyond the canonical ones for TriangleOptic:
    1) Set the faces in from_mesh.  Only has to be done when the optic is rebuilt.
    2) Apply constraints.
    3) Set the vertices from the zero points, vectors, and parameters, via get_vertices_from_params()
    4) Separate the vertices out by face via gather_faces()
    5) Apply the vertex update map via try_vum()
    4) Compute the norm via compute_norm().
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
        self.fixed_points = None
        self.gather_vertices = None
        self.gather_params = None
        self.initials = None
        self.any_fixed = False
        self.any_driven = False
        self.driver_indices = None
        self.full_params = False
        self._qt_compat = qt_compat
        self._d_mat = None
        self.movable_indices = None
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
            accumulator_active=True
        )

        if (self.enable_vum or self.enable_accumulator) and self.driver.driver_type == "client":
            self.mt_controller = cw.MeshTricksController(self, driver)
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

        # At this point we should be guarenteed to have three sets of mutually exclusive indices, that all index into
        # mesh.points:
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

        self.movable_indices = np.setdiff1d(np.arange(mesh.points.shape[0]), fixed_indices)
        movable_count = len(self.movable_indices)
        param_count = len(driver_indices)
        self.fixed_points = tf.constant(mesh.points[fixed_indices], dtype=tf.float64)
        self.zero_points = tf.constant(mesh.points[self.movable_indices], dtype=tf.float64)

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

            drivens_driver = {reindex_driven[key]: reindex_driver[value] for key, value in drivens_driver.items()}

            # Now need to construct the gather_params, a set of indices that can be used to gather from parameters to
            # zero points.
            self.gather_params = tf.constant([drivens_driver[key] for key in range(movable_count)], dtype=tf.int32)
            self._foo = [reindex_driver[v] for v in self.driver_indices]
        else:
            self._foo = self.driver_indices

        if self.any_fixed:
            # Now need to construct gather_vertices, a set of indices that can be used to gather from
            # the concatenation of fixed_points and zero_points (in that order) to the indices.
            gather_vertices = np.argsort(np.concatenate((fixed_indices, self.movable_indices)))
            self.gather_vertices = tf.constant(gather_vertices, dtype=tf.int32)

        # Make the final components of the surface.
        if initials is None:
            self.initials = tf.zeros((param_count, 1), dtype=tf.float64)
        else:
            self.initials = initials
        self.parameters = tf.Variable(self.initials)
        self.faces = tf.convert_to_tensor(self.try_flip_norm(mt.unpack_faces(mesh.faces)), dtype=tf.int32)
        self.vectors = tf.convert_to_tensor(self.vector_generator(self.zero_points), dtype=tf.float64)

        # Need to temporarily disable the vum so we can update so we can get the parts we need to make the new VUM.
        enable_vum = self.enable_vum
        self.enable_vum = False
        self.update()
        self.enable_vum = enable_vum
        self.try_mesh_tools()

    def try_mesh_tools(self):
        vertices = self.get_vertices_from_params(tf.zeros_like(self.parameters))
        if self.enable_vum:
            self.vum = tf.constant(mt.cosine_vum(
                self.settings.vum_origin,
                tuple(each.numpy() for each in self.face_vertices),
                self.vertices.numpy(),
                self.faces.numpy()
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
        if not self.frozen or force:
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
            # Compute the norm
            self.compute_norm()

    def get_vertices_from_params(self, params):
        if self.any_driven:
            # If any vertices are driven, we need expand the parameters to match the shape of the zero points.
            gathered_params = tf.gather(params, self.gather_params, axis=0)
        else:
            gathered_params = params
        vertices = self.zero_points + gathered_params * self.vectors

        if self.any_fixed:
            # If any vertices are fixed vertices, we need to add them in using gather_vertices
            all_vertices = tf.concat((self.fixed_points, vertices), axis=0)
            vertices = tf.gather(all_vertices, self.gather_vertices, axis=0)
        return vertices

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
        # Normalize, so that the average position of the surface is conserved.  I... am really unsure which axis
        # we need to normalize over.
        smoother /= tf.reduce_sum(smoother, axis=0)
        return smoother

    def smooth(self, smoother):
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


class ParamsUpdatedSignal(qtc.QObject):
    sig = qtc.pyqtSignal()


class TrigBoundaryDisplayController(qtw.QWidget):
    _valid_colors = set(pv.hexcolors.keys())

    def __init__(self, component, plot):
        super().__init__()
        self.component = component
        self._permit_draws = False
        self._params_valid = hasattr(self.component, "parameters")

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


class ClipConstraint(qtw.QWidget):
    def __init__(self, settings, min, max):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(clip_constraint_min=min, clip_constraint_max=max)
        layout = qtw.QHBoxLayout()
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


class SpacingConstraint(qtw.QWidget):
    def __init__(self, settings, min, target=None, mode="min"):
        super().__init__()
        self.settings = settings
        self.target=target
        if mode == "min":
            self._reduce = tf.reduce_min
        elif mode == "max":
            self._reduce = tf.reduce_max
        else:
            raise ValueError(f"FixedMinDistanceConstraint: mode must be 'min' or 'max'.")
        self.settings.establish_defaults(distance_constraint=min)
        layout = qtw.QHBoxLayout()
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "distance_constraint", float, validator=qtg.QDoubleValidator(-1e6, 1e6, 9)
        ))
        self.setLayout(layout)

    def __call__(self, component):
        if self.target is None:
            target = tf.zeros_like(component.parameters)
        else:
            target = self.target.parameters
        diff = self._reduce(component.parameters - target - self.settings.distance_constraint)
        diff = tf.broadcast_to(diff, component.parameters.shape)
        component.param_assign_sub(diff)
