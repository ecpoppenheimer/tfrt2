import math
import traceback

import numpy as np
import pyvista as pv
import tensorflow as tf
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

import tfrt2.component_widgets as cw
import tfrt2.mesh_tools as mt

SQRT2PI = math.sqrt(2 * math.pi)


class ParameterControls(qtw.QWidget):
    slider_ticks = 1024
    slider_ticks_half = 512

    def __init__(self, parent):
        super().__init__()
        self.parent_client = parent
        self._added_widgets = []
        self.parameter_controllers = []
        self.selectable_component = None
        self.selected_vertices = set()
        self.selected_mask = None
        self.movable_indices = set()
        self.transform_center = np.zeros((3,), dtype=np.float64)
        self.selected_mesh = None
        self._selected_mesh_actor = None
        self.component_selector = qtw.QComboBox()
        self.distance_table = None
        self.neighbor_table = None
        self.start_p = None
        self.delta_p = None
        self.slider_scale_max = 1.0
        self.slider_scale_min = 0.0
        self.consume_change = False
        self.undo_stack = qtw.QUndoStack()
        self.old_parameters = None

        self.choose_button_group = qtw.QButtonGroup()
        self.choose_singles = qtw.QRadioButton("Single")
        self.choose_center = qtw.QRadioButton("Center")
        self.choose_neighbors = qtw.QRadioButton("Neighbor")
        self.choose_distances = qtw.QRadioButton("Distance")
        self.choose_button_group.addButton(self.choose_singles)
        self.choose_button_group.addButton(self.choose_center)
        self.choose_button_group.addButton(self.choose_neighbors)
        self.choose_button_group.addButton(self.choose_distances)
        self.slider_scale_factor = qtw.QLineEdit("1.0")

        self.neighbor_count_entry = qtw.QLineEdit("1")
        self.distance_min_entry = qtw.QLineEdit("0.0")
        self.distance_max_entry = qtw.QLineEdit("1.0")
        self.adjustment_slider = cw.DelayedSlider()

        self.mode_button_group = qtw.QButtonGroup()
        self.mode_flat = qtw.QRadioButton("Flat")
        self.mode_gaussian = qtw.QRadioButton("Gaussian")
        self.mode_button_group.addButton(self.mode_flat)
        self.mode_button_group.addButton(self.mode_gaussian)
        self.gaussian_width = qtw.QLineEdit("0.01")

        self.undo_button = qtw.QPushButton("Undo")
        self.redo_button = qtw.QPushButton("Redo")

        self.rough_discrimination_factor = qtw.QLineEdit("0.85")

        self.main_layout = qtw.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Controls for selecting vertices
        select_layout = qtw.QGridLayout()
        select_layout.setContentsMargins(11, 0, 0, 0)
        self.main_layout.addWidget(qtw.QLabel(
            "Selection controls"
        ))
        self.main_layout.addLayout(select_layout)
        self.build_selection_ui(select_layout)

        self.main_layout.addStretch()

    def build_selection_ui(self, select_layout):
        ui_row = 0

        # Combo box to select which component is active
        self.component_selector.addItem("None")
        self.component_selector.currentIndexChanged.connect(self.selector_changed)
        select_layout.addWidget(self.component_selector, ui_row, 0, 1, 3)

        # Buttons that affect the whole selection
        all_button = qtw.QPushButton("All")
        all_button.clicked.connect(self.click_select_all)
        select_layout.addWidget(all_button, ui_row, 3, 1, 3)

        none_button = qtw.QPushButton("Clear")
        none_button.clicked.connect(self.click_select_none)
        select_layout.addWidget(none_button, ui_row, 6, 1, 3)

        invert_button = qtw.QPushButton("Invert")
        invert_button.clicked.connect(self.click_select_invert)
        select_layout.addWidget(invert_button, ui_row, 9, 1, 2)
        ui_row += 1

        # Buttons to toggle the selection brush
        label = qtw.QLabel(
            "Single selects a single point, Xform center selects the center of the soft transform operation.  By "
            "neighbor and by distance select points into the set be either adjacency to the selected point or "
            "distance.  Selection controls operate relative to the base mesh of the optic."
        )
        label.setWordWrap(True)
        select_layout.addWidget(label, ui_row, 0, 1, 12)
        ui_row += 1
        select_layout.addWidget(self.choose_singles, ui_row, 0, 1, 3)
        select_layout.addWidget(self.choose_center, ui_row, 3, 1, 3)
        select_layout.addWidget(self.choose_neighbors, ui_row, 6, 1, 3)
        select_layout.addWidget(self.choose_distances, ui_row, 9, 1, 3)
        self.choose_singles.setChecked(True)
        ui_row += 1

        # Line edit to get the number of neighbors
        self.neighbor_count_entry.setValidator(qtg.QIntValidator(1, 100))
        self.neighbor_count_entry.setMaximumWidth(35)
        select_layout.addWidget(qtw.QLabel("Neighbors:"), ui_row, 0, 1, 3)
        select_layout.addWidget(self.neighbor_count_entry, ui_row, 3, 1, 1)

        # Line edits to get the distance range
        select_layout.addWidget(qtw.QLabel("Min"), ui_row, 4, 1, 1)
        self.distance_min_entry.setValidator(qtg.QDoubleValidator(0, 1000, 6))
        select_layout.addWidget(self.distance_min_entry, ui_row, 5, 1, 3)
        self.distance_max_entry.setValidator(qtg.QDoubleValidator(0, 1000, 6))
        select_layout.addWidget(qtw.QLabel("Max"), ui_row, 8, 1, 1)
        select_layout.addWidget(self.distance_max_entry, ui_row, 9, 1, 3)
        ui_row += 1

        # Set the adjustment mode
        select_layout.addWidget(qtw.QLabel("Adjustment mode"), ui_row, 0, 1, 3)
        select_layout.addWidget(self.mode_flat, ui_row, 3, 1, 2)
        self.mode_flat.setChecked(True)
        select_layout.addWidget(self.mode_gaussian, ui_row, 5, 1, 3)
        self.gaussian_width.setValidator(qtg.QDoubleValidator(1e-6, 1e3, 6))

        # Set the width of the gaussian adjustment
        select_layout.addWidget(qtw.QLabel("Width"), ui_row, 8, 1, 1)
        select_layout.addWidget(self.gaussian_width, ui_row, 9, 1, 3)
        ui_row += 1
        select_layout.addWidget(qtw.QLabel("Speed"), ui_row, 0, 1, 2)
        self.slider_scale_factor.setValidator(qtg.QDoubleValidator(1e-6, 1e6, 4))
        select_layout.addWidget(self.slider_scale_factor, ui_row, 2, 1, 3)

        # Place undo/redo buttons
        self.undo_stack.canRedoChanged.connect(self.set_redo_state)
        self.undo_stack.canUndoChanged.connect(self.set_undo_state)

        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)
        select_layout.addWidget(self.undo_button, ui_row, 6, 1, 3)

        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)
        select_layout.addWidget(self.redo_button, ui_row, 9, 1, 3)
        ui_row += 1

        # Slider to adjust the value of the selection
        select_layout.addWidget(self.adjustment_slider, ui_row, 0, 1, 12)
        self.adjustment_slider.setMinimum(0)
        self.adjustment_slider.setMaximum(self.slider_ticks)
        self.adjustment_slider.setValue(self.slider_ticks_half)
        self.adjustment_slider.valueChanged.connect(self.slider_changed)
        self.adjustment_slider.sliderPressed.connect(self.slider_pressed)
        self.adjustment_slider.sliderReleased.connect(self.slider_released)
        ui_row += 1

        # Button to attempt to auto-flatten the selected spikey region
        label = qtw.QLabel("To automatically find rough spots, select a single point in the middle of a flat area.")
        label.setWordWrap(True)
        select_layout.addWidget(label, ui_row, 0, 1, 12)
        ui_row += 1

        auto_flat_button = qtw.QPushButton("Auto Flatten")
        auto_flat_button.clicked.connect(self.auto_flatten)
        select_layout.addWidget(auto_flat_button, ui_row, 0, 1, 3)

        # Controls to try automatically selecting rough regions of the surface
        select_rough_button = qtw.QPushButton("Find Rough Spots")
        select_rough_button.clicked.connect(self.select_rough)
        select_layout.addWidget(select_rough_button, ui_row, 3, 1, 4)

        self.rough_discrimination_factor.setValidator(qtg.QDoubleValidator(-1, 1, 4))
        select_layout.addWidget(self.rough_discrimination_factor, ui_row, 7, 1, 3)

    def add_widget(self, widget):
        self.main_layout.insertWidget(self.main_layout.count() - 1, widget)
        self._added_widgets.append(widget)

    def update_with_system(self, system):
        for widget in self._added_widgets:
            if hasattr(widget, "remove_drawer"):
                widget.remove_drawer()
            widget.deleteLater()
        self._added_widgets = []
        self.component_selector.setCurrentIndex(0)
        for i in range(self.component_selector.count() - 1, 0, -1):
            self.component_selector.removeItem(i)

        if system is not None:
            for part in system.parametric_optics:
                controller = ParameterController(part, self.parent_client)
                self.add_widget(qtw.QLabel(part.name))
                self.add_widget(controller)
                self.parameter_controllers.append(controller)
                self.component_selector.addItem(part.name)

    def refresh_parameters(self):
        for each in self.parameter_controllers:
            each.refresh_parameters()

    def point_picked(self, component, point):
        """
        Component will be a parametric optic, or None, in which case the self.selected_mesh was selected, in which
        case we should deselect a point.
        """
        if component is None:
            # Selected a point on the selection mesh.  I thought this would happen but am not able to trigger this
            # condition.  But if it ever does happen,  point will index into the selected_indices, rather than the
            # component vertices.
            print(f"just selected a point on the selection mesh - how?")
            # self.process_selection(self.selected_indices[point], False)
        elif component is self.selectable_component:
            self.process_selection(point, point not in self.selected_vertices)
        self.update_selection()

    def process_selection(self, p_index, add):
        """
        Determine which points to add to / remove from the collection when a point, p_index, is clicked.

        Already know that the component is self.selectable_component.

        Parameters
        ----------
        p_index : int
            The index (of points in the component) of the point being interacted with by the user.
        add : bool
            If True, this is a positive selection (want to add to the collection).  If False, this is a
            negative selection (want to remove to the collection).
        """
        if self.choose_singles.isChecked():
            selected = p_index
        elif self.choose_center.isChecked():
            self.transform_center = self.selectable_component.tiled_base_mesh.points[p_index]
            selected = set()
        elif self.choose_neighbors.isChecked():
            #selected = set((p_index,))
            #points_to_check = set((p_index,))
            selected = {p_index}
            points_to_check = {p_index}
            for _ in range(int(self.neighbor_count_entry.text())):
                new_neighbors = set()
                for p in points_to_check:
                    new_neighbors |= self.neighbor_table[p] - selected
                selected |= new_neighbors
                points_to_check = new_neighbors
        else:
            distance = self.distance_table[p_index]
            min_d, max_d = float(self.distance_min_entry.text()), float(self.distance_max_entry.text())
            all_indices = np.arange(self.selectable_component.vertices.shape[0])
            mask = np.logical_and(min_d <= distance, distance <= max_d)
            selected = set(all_indices[mask])

        if add:
            self.selected_vertices |= self.filter(selected)
        else:
            try:
                self.selected_vertices -= selected
            except TypeError:
                self.selected_vertices.discard(selected)

    def update_selection(self, avoid_recursion=False):
        try:
            if self.selectable_component is None:
                return
            if self.selectable_component.p_controller_ack_remeshed:
                self.handle_remesh()

            self.parent_client.plot.remove_actor(self._selected_mesh_actor)
            if len(self.selected_vertices) > 0:
                all_points = self.selectable_component.vertices.numpy()
                self.selected_mesh = pv.PolyData(all_points[list(self.selected_vertices)])
                self._selected_mesh_actor = self.parent_client.plot.add_mesh(
                    self.selected_mesh, render_points_as_spheres=True, color="yellow", point_size=10.0
                )
            self.selected_mask = np.array(tuple(
                1.0 if i in self.selected_vertices else 0.0
                for i in range(self.selectable_component.tiled_base_mesh.points.shape[0])
            ))
        except IndexError: # Attempt to fix a crash when the system is reloaded while vertices are selected, I think.
            if self.selectable_component is not None and not avoid_recursion:
                self.selected_vertices = set()
                self.update_selection(True)

    def selector_changed(self, _):
        name = self.component_selector.currentText()
        self.selected_vertices = set()
        self.selected_mesh = None
        self.parent_client.plot.remove_actor(self._selected_mesh_actor)
        if name == "None":
            self.selectable_component = None
            self.movable_indices = set()
            self.distance_table = None
            self.neighbor_table = None
        else:
            self.selectable_component = self.parent_client.optical_system.parts[name]
            self.movable_indices = set(self.selectable_component.movable_indices)
            self.distance_table = mt.mesh_distance_matrix(self.selectable_component.tiled_base_mesh)
            self.neighbor_table = mt.mesh_neighbor_table(self.selectable_component.tiled_base_mesh)

    def filter(self, points):
        try:
            p = set(points)
        except TypeError:
            p = {points}
        return p & self.movable_indices

    def click_select_all(self):
        if self.selectable_component is not None:
            self.selected_vertices = self.filter(np.arange(self.selectable_component.vertices.shape[0]))
            self.update_selection()

    def click_select_none(self):
        if self.selectable_component is not None:
            self.selected_vertices = set()
            self.update_selection()

    def click_select_invert(self):
        if self.selectable_component is not None:
            self.selected_vertices = self.filter(np.arange(self.selectable_component.vertices.shape[0])) - \
                 self.selected_vertices
            self.update_selection()

    def slider_pressed(self):
        if self.selectable_component is None or len(self.selected_vertices) == 0:
            return
        if self.selectable_component.p_controller_ack_remeshed:
            self.handle_remesh()

        self.start_p = self.selectable_component.parameters.numpy()
        self.slider_scale_max = np.amax(np.abs(self.start_p)) * 2
        self.slider_scale_min = min(np.amin(np.abs(self.start_p)) * 2, -self.slider_scale_max)
        self.old_parameters = self.selectable_component.parameters.numpy()

    def slider_released(self):
        if self.selectable_component is None or len(self.selected_vertices) == 0:
            return

        self.start_p = None
        self.delta_p = None
        self.consume_change = True
        self.adjustment_slider.setValue(self.slider_ticks_half)

        self.undo_stack.push(ParamUpdateAction(
            self.selectable_component,
            self,
            self.selectable_component.parameters.numpy(),
            self.old_parameters,
        ))

    def slider_changed(self, value):
        if self.consume_change:
            self.consume_change = False
            return
        if self.selectable_component is None or len(self.selected_vertices) == 0:
            return
        if self.start_p is None:
            self.slider_pressed()

        # Determine the amount each vertex should change
        if self.mode_flat.isChecked():
            delta_v = self._rescale_slider(value) * self.selected_mask
        elif self.mode_gaussian.isChecked():
            all_v_distance = np.linalg.norm(
                np.reshape(self.transform_center, (1, 3)) - self.selectable_component.tiled_base_mesh.points,
                axis=1
            )
            sigma = float(self.gaussian_width.text())
            delta_v = self._rescale_slider(value) * np.exp(-(all_v_distance / sigma)**2 / 2) / (sigma * SQRT2PI)
        else:
            raise RuntimeError("Parameter Controller: got into state where neither adjustment mode is selected.")

        # Convert the delta_v to delta_p via component.v_to_p.
        # Each parameter could be affected by more than one vertex.  How to handle this?  At first I decided to
        # average, but that turns out to be super ugly when adjusting across a symmetry plane, as the adjustment gets
        # doubled.  So instead I am going to select and keep only the maximum magnitude adjustment.
        self.delta_p = np.zeros((self.selectable_component.parameters.shape[0],), dtype=np.float64)
        for v in self.selected_vertices:
            if abs(delta_v[v]) > abs(self.delta_p[self.selectable_component.v_to_p[v]]):
                self.delta_p[self.selectable_component.v_to_p[v]] = delta_v[v]

        # Update the component and display
        self.selectable_component.param_assign(self.start_p + np.reshape(self.delta_p, (-1, 1)))
        self.selectable_component.update()
        self.selectable_component.redraw()
        self.update_selection()

    def _rescale_slider(self, v):
        rtn = v * float(self.slider_scale_factor.text()) * (self.slider_scale_max - self.slider_scale_min) / \
            self.slider_ticks + float(self.slider_scale_factor.text()) * self.slider_scale_min
        return rtn

    def set_undo_state(self, state):
        self.undo_button.setEnabled(state)

    def set_redo_state(self, state):
        self.redo_button.setEnabled(state)

    def undo(self):
        self.undo_stack.undo()

    def redo(self):
        self.undo_stack.redo()

    def auto_flatten(self):
        if self.selectable_component is None or len(self.selected_vertices) == 0:
            return
        if len(self.selected_vertices) == self.selectable_component.tiled_base_mesh.points.shape[0]:
            print("Cannot flatten with all vertices selected!")
            return
        start_p = self.selectable_component.parameters.numpy()
        new_parameters = start_p.copy()

        # Only want to process once for each parameter, so Find the set of unique parameters associated with these
        # vertices.
        selected_list = np.array(list(self.selected_vertices), dtype=np.int32)
        selected_p_per_v = self.selectable_component.v_to_p[selected_list]
        unique_parameters = set(selected_p_per_v)

        # selected_p_per_v is a numpy array that contains each parameter, for each selected vertex.
        # Need to construct a reverse map from unique_parameters to these selected vertices
        p_to_v = {p: set() for p in unique_parameters}
        for v, p in zip(selected_list, selected_p_per_v):
            p_to_v[p].add(v)

        # Go through selected_vertices, ignoring ones whose parameter has already been visited
        working_vertices = self.selected_vertices.copy()
        while working_vertices:
            # Count how many nonworking neighbors each selected vertex has.  I really wish I could have done this
            # once but the number changes every time working vertices are removed, so have to re-compute every cycle.
            # Ok, I am sure there is a very complicated solution, but I don't think it is worth trying to find it.
            working_list = list(working_vertices)
            neighbors_count = [len(self.neighbor_table[v] - working_vertices) for v in working_list]
            v = working_list[np.argmax(neighbors_count)]

            working_vertices.discard(v)
            p = self.selectable_component.v_to_p[v]
            if p in unique_parameters:
                # Only visit each parameter once
                unique_parameters.discard(p)

                # These neighbors are adjacent to v, unselected, and have not still pending an update
                unworking_neighbors = list(self.neighbor_table[v] - working_vertices)
                new_parameters[p] = np.mean(new_parameters[self.selectable_component.v_to_p[unworking_neighbors]])

                # Already removed v from working_vertices, now need to remove any other vertices that share
                # a parameter with it
                working_vertices -= p_to_v[p]

            if not working_vertices:
                break

        # All parameters have been moved, so push the update
        self.undo_stack.push(ParamUpdateAction(
            self.selectable_component,
            self,
            new_parameters,
            start_p,
        ))

    def select_rough(self):
        if self.selectable_component is None:
            return
        if len(self.selected_vertices) != 1:
            print("Please select exactly one vertex in a flat region to use as the starting point.")
            return

        # Create a map from edges to faces, so we can quickly move between adjacent faces.
        # I will refer to faces by index, since I need to get the face norms, which are already computed and stored
        # by the optic.  In the code that follows, 'f' refers the index of a face and 'face' refers to a set of the
        # vertices that face contains.  I will need both
        edge_to_face = {}
        starting_face = None
        starting_vertex = self.selected_vertices.pop()
        faces = mt.unpack_faces(self.selectable_component.tiled_base_mesh.faces)
        for f, face in zip(range(faces.shape[0]), faces):
            for i1, i2 in ((0, 1), (1, 2), (2, 0)):
                edge = frozenset({face[i1], face[i2]})
                try:
                    # If the below index works, then there has already been (hopefully exactly one) face associated
                    # with this edge, so replace it with a set of that one and this one.
                    # Each edge should only ever have either one or two faces.
                    prev_face = edge_to_face[edge]
                    edge_to_face[edge] = {prev_face, f}
                except KeyError:
                    # If we get here, this is the first face to be associated with this edge, so add it as a single
                    # int.
                    edge_to_face[edge] = f
            if starting_face is None:
                if starting_vertex in face:
                    starting_face = f

        # Starting from the starting face, go through connected faces, check that the norms are not too different,
        # and if not, add the new faces vertices to the flat selection and add that face to the stack.  This is
        # a depth-first search.
        norm_threshold = float(self.rough_discrimination_factor.text())
        visited_faces = set()
        face_stack = [starting_face]
        flat_selection = set()
        while face_stack:
            f = face_stack.pop()
            face = faces[f]
            visited_faces.add(f)
            norm = self.selectable_component.norm[f]
            flat_selection |= set(face)
            for i1, i2 in ((0, 1), (1, 2), (2, 0)):
                edge = frozenset({face[i1], face[i2]})

                # Get the adjacent face associated with this edge.  edge_to_face[edge] contains either a single int,
                # in which case this edge is an edge of the mesh, and has no corresponding other face.  In this case
                # there is nothing to do.  The other case is that edge_to_face[edge] is a set of two elements, the
                # current face and one other, so just need to get that other face
                candidate_faces = edge_to_face[edge]
                if type(candidate_faces) is int:
                    continue
                adjacent_face = (candidate_faces - {f}).pop()
                if adjacent_face in visited_faces:
                    # Need to not process faces more than once
                    continue
                adjacent_norm = self.selectable_component.norm[adjacent_face]

                # Optics are supposed to normalize their norms, so don't need to worry about that here.
                if mt.np_dot(norm, adjacent_norm) > norm_threshold:
                    face_stack.append(adjacent_face)

        # Select all vertices not in flat selection
        self.selected_vertices = self.filter(np.arange(self.selectable_component.vertices.shape[0])) - flat_selection
        self.update_selection()

    def handle_remesh(self):
        self.selectable_component.p_controller_ack_remeshed = False
        self.click_select_none()
        self.selector_changed(None)

    def undoable_parameter_update(self, component, new_params, old_params=None):
        if old_params is None:
            old_params = component.parameters
        self.undo_stack.push(ParamUpdateAction(component, self, new_params, old_params))


class ParamUpdateAction(qtw.QUndoCommand):
    def __init__(self, component, param_controller, new_params, old_params):
        self.component = component
        self.param_controller = param_controller
        self.new_params = np.array(new_params, dtype=np.float64)
        self.old_params = np.array(old_params, dtype=np.float64)
        super().__init__()

    def redo(self):
        try:
            self.component.param_assign(self.new_params)
            self.component.update()
            self.component.redraw()
            self.param_controller.update_selection()
        except ValueError:
            print("Could not redo - parameter shape mismatch")

    def undo(self):
        try:
            self.component.param_assign(self.old_params)
            self.component.update()
            self.component.redraw()
            self.param_controller.update_selection()
        except ValueError:
            print("Could not redo - parameter shape mismatch")


class ParameterController(qtw.QWidget):
    def __init__(self, component, client):
        super().__init__()
        self.component = component
        self.client = client
        self._suppress_updates = False
        self.smoother = None

        # Build the UI elements
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(11, 11, 0, 11)
        self.setLayout(main_layout)

        # Parameter list
        show_params = qtw.QCheckBox("Show parameters")
        show_params.setCheckState(False)
        show_params.setTristate(False)
        main_layout.addWidget(show_params, 0, 0)

        def refresh_click():
            self.refresh_parameters()

        refresh_list_button = qtw.QPushButton("Refresh")
        refresh_list_button.hide()
        refresh_list_button.clicked.connect(refresh_click)
        main_layout.addWidget(refresh_list_button, 0, 1)

        self.parameter_list = qtw.QTableWidget(0, 1)
        self.parameter_list.setHorizontalHeaderLabels(("Index", "Value"))
        self.parameter_list.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.parameter_list.horizontalHeader().setSectionResizeMode(0, qtw.QHeaderView.Stretch)
        self.parameter_list.hide()
        self.parameter_list.itemChanged.connect(self.edit_parameter)
        main_layout.addWidget(self.parameter_list, 1, 0, 1, 2)

        def toggle_list():
            state = not self.parameter_list.isHidden()
            if state:
                self.parameter_list.hide()
                refresh_list_button.hide()
            else:
                self.parameter_list.show()
                refresh_list_button.show()
                self.refresh_parameters()

        show_params.clicked.connect(toggle_list)

        # Button to reset the parameters
        reset_widget = qtw.QWidget()
        reset_layout = qtw.QHBoxLayout()
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_widget.setLayout(reset_layout)
        reset_button = qtw.QPushButton("Reset to initials")

        def click_reset():
            try:
                self.client.parameters_pane.undoable_parameter_update(self.component, self.component.initials)
                self.update_everything()
            except Exception:
                print(f"Could not reset parameters:")
                print(traceback.format_exc())

        reset_button.clicked.connect(click_reset)
        reset_layout.addWidget(reset_button)

        noise_button = qtw.QPushButton("Add Noise")
        noise_scale = qtw.QLineEdit(".1")
        noise_scale.setValidator(qtg.QDoubleValidator(1e-6, 1e3, 4))

        def click_noise():
            try:
                scale = float(noise_scale.text())
                new_params = tf.random.normal(self.component.parameters.shape, stddev=scale, dtype=tf.float64) +\
                    self.component.parameters
                self.client.parameters_pane.undoable_parameter_update(self.component, new_params)
                self.update_everything()
            except Exception:
                print(f"Could not add noise to parameters:")
                print(traceback.format_exc())

        noise_button.clicked.connect(click_noise)
        reset_layout.addWidget(noise_button)
        reset_layout.addWidget(noise_scale)

        main_layout.addWidget(reset_widget, 3, 0, 1, 2)

        # Button to test the accumulator on a parameter.
        try:
            if self.component.accumulator is not None:
                acumulator_widget = qtw.QWidget()
                acumulator_layout = qtw.QGridLayout()
                acumulator_layout.setContentsMargins(0, 0, 0, 0)
                acumulator_widget.setLayout(acumulator_layout)

                acumulator_layout.addWidget(qtw.QLabel("Test Accumulator"), 0, 0)
                acumulator_test_button = qtw.QPushButton("Test")
                acumulator_test_button.clicked.connect(self.test_acumulator)
                acumulator_layout.addWidget(acumulator_test_button, 1, 0)

                acumulator_layout.addWidget(qtw.QLabel("Vertex"), 0, 1)
                self.acumulator_vertex_edit = qtw.QLineEdit("0")
                self.acumulator_vertex_edit.setValidator(qtg.QIntValidator(0, self.component.parameters.shape[0] - 1))
                acumulator_layout.addWidget(self.acumulator_vertex_edit, 1, 1)

                acumulator_layout.addWidget(qtw.QLabel("Adjustment"), 0, 2)
                self.acumulator_scale_edit = qtw.QLineEdit(".05")
                self.acumulator_scale_edit.setValidator(qtg.QDoubleValidator(1e-9, 1e-9, 10))
                acumulator_layout.addWidget(self.acumulator_scale_edit, 1, 2)

                main_layout.addWidget(acumulator_widget, 5, 0, 1, 2)
        except Exception:
            pass

        # Make a smoother
        main_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "smooth_stddev", float, qtg.QDoubleValidator(1e-6, 1e6, 8), self.update_smoother
        ), 6, 0, 1, 1)
        main_layout.addWidget(cw.SettingsCheckBox(
            self.component.settings, "smooth_active", "Active"
        ), 6, 1, 1, 1)
        smooth_button = qtw.QPushButton("Test Smoother")
        smooth_button.clicked.connect(self.test_smoother)
        main_layout.addWidget(smooth_button, 7, 0, 1, 1)

        # Entry box for the relative LR
        main_layout.addWidget(cw.SettingsEntryBox(
            self.component.settings, "relative_lr", float, qtg.QDoubleValidator(0, 1e10, 8), self.update_smoother
        ), 8, 0, 1, 2)

        # optionally register a callback to update the parameters from the boundary.  This only works if the boundary
        # explicitly calls the signal
        try:
            self.component.parameters_updated.sig.connect(self.refresh_parameters)
        except Exception:
            pass

    def test_acumulator(self):
        vertex = int(self.acumulator_vertex_edit.text())
        if vertex >= self.component.parameters.shape[0]:
            vertex = self.component.parameters.shape[0] - 1
            self.acumulator_vertex_edit.setText(str(vertex))

        adjustment = float(self.acumulator_scale_edit.text())
        delta = np.zeros_like(self.component.parameters)
        delta[vertex] = adjustment

        delta = self.component.try_accumulate(delta)

        new_params = self.component.parameters + delta
        self.client.parameters_pane.undoable_parameter_update(self.component, new_params)
        self.update_everything()

    def refresh_parameters(self):
        try:
            count = self.component.parameters.shape[0]
            values = [f"{v[0]:.10f}" for v in self.component.parameters.numpy()]
            self._suppress_updates = True
            if self.parameter_list.rowCount() != count:
                self.parameter_list.setRowCount(count)
                self.parameter_list.setVerticalHeaderLabels((str(i) for i in range(count)))
                for i, v in zip(range(count), values):
                    self.parameter_list.setItem(i, 0, qtw.QTableWidgetItem(v))

                # since we have detected a change in the parameter count, update the acumulator vertex validator
                try:
                    self.acumulator_vertex_edit.setValidator(qtg.QIntValidator(0, self.component.parameters.shape[0] - 1))
                except Exception:
                    pass
            else:
                for i, v in zip(range(count), values):
                    self.parameter_list.item(i, 0).setText(v)
            self._suppress_updates = False
        except Exception:
            pass

    def edit_parameter(self, item):
        if self._suppress_updates:
            return
        vertex = item.row()
        params = self.component.parameters.numpy()
        params[vertex] = float(item.text())
        self.client.parameters_pane.undoable_parameter_update(self.component, params)
        self.update_everything()

    def update_everything(self):
        # component.update respects constraints, which can hide changes, so lets re-update this item to reflect that...
        # Nope, because constraints can be more complicated than that.  Update the entire display.
        self.client.update_optics()
        self.client.try_auto_retrace()
        self.refresh_parameters()
        try:
            self.component.drawer.draw()
        except AttributeError:
            pass

    def update_smoother(self):
        self.component.smoother = self.component.get_smoother(self.component.settings.smooth_stddev)

    def test_smoother(self):
        old_params = self.component.parameters.numpy()
        self.component.smooth()
        new_params = self.component.parameters.numpy()
        self.client.parameters_pane.undoable_parameter_update(self.component, new_params, old_params)
        self.update_everything()
