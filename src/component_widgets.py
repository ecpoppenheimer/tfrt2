import traceback
import pathlib
import pickle

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import pyvista as pv
import numpy as np
import tensorflow as tf

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class SettingsEntryBox(qtw.QWidget):
    def __init__(
            self, settings, key, value_type, validator=None, callback=None, label=None
    ):
        super().__init__()
        layout = qtw.QHBoxLayout()
        self.setLayout(layout)

        if label is None:
            label = qtw.QLabel(str(key).replace("_", " "))
        else:
            label = qtw.QLabel(str(label))
        label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)
        layout.addWidget(label)

        edit_box = qtw.QLineEdit()
        layout.addWidget(edit_box)
        edit_box.setText(str(settings.dict[key]))
        if validator:
            edit_box.setValidator(validator)

        def edit_callback():
            value = value_type(edit_box.text())
            settings.dict[key] = value

        edit_box.editingFinished.connect(edit_callback)
        if callback is not None:
            try:
                for each in callback:
                    edit_box.editingFinished.connect(each)
            except TypeError:
                edit_box.editingFinished.connect(callback)


class SettingsRangeBox(qtw.QWidget):
    def __init__(self, settings, label, low_key, high_key, value_type, validator=None, callback=None):
        super().__init__()
        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        self.settings = settings

        label = qtw.QLabel(label)
        label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)
        layout.addWidget(label)

        low_entry = qtw.QLineEdit()
        layout.addWidget(low_entry)
        low_entry.setText(str(settings.dict[low_key]))
        if validator:
            low_entry.setValidator(validator)

        high_entry = qtw.QLineEdit()
        layout.addWidget(high_entry)
        high_entry.setText(str(settings.dict[high_key]))
        if validator:
            high_entry.setValidator(validator)

        def low_callback():
            low_value = value_type(low_entry.text())
            high_value = value_type(high_entry.text())

            if low_value >= high_value:
                low_entry.setStyleSheet("QLineEdit { background-color: pink}")
            else:
                low_entry.setStyleSheet("QLineEdit { background-color: white}")
                settings.dict[low_key] = low_value
                if callback is not None:
                    try:
                        for each in callback:
                            each()
                    except TypeError:
                        callback()

        def high_callback():
            low_value = value_type(low_entry.text())
            high_value = value_type(high_entry.text())

            if high_value <= low_value:
                high_entry.setStyleSheet("QLineEdit { background-color: pink}")
            else:
                high_entry.setStyleSheet("QLineEdit { background-color: white}")
                settings.dict[high_key] = high_value
                if callback is not None:
                    try:
                        for each in callback:
                            each()
                    except TypeError:
                        callback()
        low_entry.editingFinished.connect(low_callback)
        high_entry.editingFinished.connect(high_callback)


class SettingsFileBox(qtw.QWidget):
    def __init__(
            self, settings, key, system_path, filter="*", mode=None, callback=None, callback_2=None
    ):
        super().__init__()
        self.filter = filter
        self.system_path = system_path
        if mode == "save":
            self.do_save = True
            self.do_load = False
            self.do_save_dialog_type = True
            self.save_callback = callback
            self.load_callback = None
        elif mode == "load":
            self.do_save = False
            self.do_load = True
            self.do_save_dialog_type = False
            self.save_callback = None
            self.load_callback = callback
        elif mode == "both":
            self.do_save = True
            self.do_load = True
            self.do_save_dialog_type = True
            self.save_callback = callback
            self.load_callback = callback_2
        elif mode == "none" or mode is None:
            self.do_save = False
            self.do_load = False
            self.do_save_dialog_type = True
            self.save_callback = None
            self.load_callback = None
        else:
            raise ValueError("SettingsFileBox: mode must be specified, and one of {save, load, both, none}")
        self.settings = settings
        self.key = key

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)

        if self.do_save:
            self.save_button = qtw.QPushButton("Save")
            self.save_button.setMaximumWidth(32)
            self.save_button.clicked.connect(self.save)
            layout.addWidget(self.save_button)
        if self.do_load:
            self.load_button = qtw.QPushButton("Load")
            self.load_button.setMaximumWidth(32)
            self.load_button.clicked.connect(self.load)
            layout.addWidget(self.load_button)

        self.select_button = qtw.QPushButton("Select")
        self.select_button.setMaximumWidth(37)
        self.select_button.clicked.connect(self.select)
        layout.addWidget(self.select_button)

        self.label = qtw.QLineEdit()
        self.label.setText(str(self.settings.dict[key]))
        self.label.setReadOnly(True)
        layout.addWidget(self.label)

    def save(self):
        if self.save_callback is not None:
            try:
                self.save_callback()
            except TypeError:
                for each in self.save_callback:
                    each()

    def load(self):
        if self.load_callback is not None:
            try:
                self.load_callback()
            except TypeError:
                for each in self.load_callback:
                    each()

    def select(self):
        if self.do_save:
            selected_file, _ = qtw.QFileDialog.getSaveFileName(
                directory=str(self.system_path), filter=self.filter
            )
        else:
            selected_file, _ = qtw.QFileDialog.getOpenFileName(
                directory=str(self.system_path), filter=self.filter
            )
        if selected_file:
            self.settings.dict[self.key] = str(pathlib.Path(selected_file))
            self.label.setText(selected_file)


class SettingsComboBox(qtw.QWidget):
    def __init__(self, component, label, settings_key, settings_options, callback=None):
        super().__init__()
        self.component = component
        self.settings_key = settings_key
        self.settings_options = settings_options

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel(label))

        selector = qtw.QComboBox()
        layout.addWidget(selector)
        selector.addItems(settings_options)
        selector.setCurrentIndex(settings_options.index(self.component.settings.dict[settings_key]))
        selector.currentIndexChanged.connect(self.set_setting)

        if callback is not None:
            try:
                for each in callback:
                    selector.currentIndexChanged.connect(each)
            except TypeError:
                selector.currentIndexChanged.connect(callback)

    def set_setting(self, index):
        self.component.settings.dict[self.settings_key] = self.settings_options[index]


class SettingsVectorBox(qtw.QWidget):
    def __init__(self, component, label, settings_key, callback=None):
        super().__init__()
        self.component = component
        self.settings_key = settings_key

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel(label))
        self.entries = []
        for i in range(3):
            initial = component.settings.dict[settings_key][i]
            entry = qtw.QLineEdit()
            self.entries.append(entry)
            entry.setText(str(initial))
            entry.setValidator(qtg.QDoubleValidator(-1e6, 1e6, 7))
            layout.addWidget(entry)

        self.entries[0].editingFinished.connect(self.callback_x)
        self.entries[1].editingFinished.connect(self.callback_y)
        self.entries[2].editingFinished.connect(self.callback_z)

        if callback is not None:
            try:
                for each in callback:
                    self.entries[0].editingFinished.connect(each)
                    self.entries[1].editingFinished.connect(each)
                    self.entries[2].editingFinished.connect(each)
            except TypeError:
                self.entries[0].editingFinished.connect(callback)
                self.entries[1].editingFinished.connect(callback)
                self.entries[2].editingFinished.connect(callback)

    def callback_x(self):
        value = float(self.entries[0].text())
        self.component.settings.dict[self.settings_key][0] = value

    def callback_y(self):
        value = float(self.entries[1].text())
        self.component.settings.dict[self.settings_key][1] = value

    def callback_z(self):
        value = float(self.entries[2].text())
        self.component.settings.dict[self.settings_key][2] = value


class SettingsCheckBox(qtw.QWidget):
    def __init__(self, component, label, settings_key, callback=None):
        super().__init__()
        self.component = component
        self.settings_key = settings_key

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel(label))

        check_box = qtw.QCheckBox()
        layout.addWidget(check_box)
        check_box.setCheckState(self.component.settings.dict[settings_key])
        check_box.setTristate(False)

        def set_setting(new_state):
            self.component.settings.dict[self.settings_key] = bool(new_state)

        check_box.stateChanged.connect(set_setting)

        if callback is not None:
            try:
                for each in callback:
                    check_box.stateChanged.connect(each)
            except TypeError:
                check_box.stateChanged.connect(callback)


class MPLWidget(qtw.QWidget):
    """
    A pyqt widget that displays a mpl plot, with a built in toolbar.

    Parameters
    ----------
    alignment : 4-tuple of floats, optional
        The area in figure coordinates where the axes will sit.  Defaults to None, in which
        case the value used is chosen based on the value of parameter blank.  If blank is
        True, uses the value (0.0, 0.0, 1.0, 1.0) in which case the axes cover exactly the
        whole drawing area, which is great if you are displaying an image.  If blank is False,
        uses (.1, .1, .898, .898) which is a good starter option for plots with displayed
        labels, but you may want to tweak manually.  .9 has a tendency to cut off the line on
        the left size of the plot.
        The first two numbers anchor the lower left corner, the second two
        are a span.  The values are all relative to the size of the canvas, so between 0 and
        1.
    fig_size : 2-tuple of floats, optional
        The default size of the figure.  But since pyqt messes with widget size a lot this
        is more of a rough starting guess than an actual set value..
    blank : bool, optional
        If False, the default, draws the plot as normal.
        If True, the canvas will be a blank white square with no axes or anything, ideal
        for drawing.
    args and kwargs passed to qtw.QWidget constructor

    Public Properties
    -----------------
    fig : mpl figure
        A handle to the generated mpl figure.
    ax : mpl axes
        A handle to the generated axes, where you can draw/plot stuff.
    fig_canvas : mpl backend object
        The canvas object that plugs into pyqt.

    Public Methods
    --------------
    draw() :
        Draw or redraw the figure.
    """

    def __init__(
            self,
            name=None,
            alignment=None,
            fig_size=(2.5, 2.5),
            blank=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.fig = Figure(figsize=fig_size)
        self.fig_canvas = FigureCanvas(self.fig)
        if alignment is None:
            if blank:
                alignment = (0.0, 0.0, 1.0, 1.0)
            else:
                alignment = (.1, .1, .898, .898)
        self.ax = self.fig.add_axes(alignment)
        if blank:
            self.ax.set_frame_on(False)
            self.ax.axes.get_xaxis().set_visible(False)
            self.ax.axes.get_yaxis().set_visible(False)
            self.ax.axes.get_xaxis().set_major_locator(plt.NullLocator())
            self.ax.axes.get_yaxis().set_major_locator(plt.NullLocator())

        layout = qtw.QVBoxLayout()
        if name:
            label = qtw.QLabel(name)
            label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Fixed)
            layout.addWidget(label)
        layout.addWidget(self.fig_canvas)
        layout.addWidget(NavigationToolbar(self.fig_canvas, self))
        self.setLayout(layout)

    def draw(self):
        self.fig_canvas.draw()


# ======================================================================================================================


class ParameterController(qtw.QWidget):
    def __init__(self, component, client):
        super().__init__()
        self.component = component
        self.client = client
        self._suppress_updates = False
        self.smoother = None

        # Build the UI elements
        main_layout = qtw.QGridLayout()
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
        reset_widget.setLayout(reset_layout)
        reset_button = qtw.QPushButton("Reset to initials")

        def click_reset():
            try:
                self.component.parameters.assign(self.component.initials)
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
                self.component.parameters.assign_add(
                    tf.random.normal(self.component.parameters.shape, stddev=scale, dtype=tf.float64)
                )
                self.update_everything()
            except Exception:
                print(f"Could not reset parameters:")
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
        self.component.settings.establish_defaults(smooth_stddev=.05)
        main_layout.addWidget(SettingsEntryBox(
            self.component.settings, "smooth_stddev", float, qtg.QDoubleValidator(1e-6, 1e6, 8), self.make_smoother
        ), 6, 0, 1, 2)
        self.make_smoother()
        smooth_button = qtw.QPushButton("Test Smoother")
        smooth_button.clicked.connect(self.test_smoother)
        main_layout.addWidget(smooth_button, 7, 0, 1, 1)

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

        self.component.parameters.assign_add(delta)
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
        self.component.parameters.assign(params)
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

    def make_smoother(self):
        self.smoother = self.component.get_smoother(self.component.settings.smooth_stddev)

    def test_smoother(self):
        if self.smoother is None:
            self.make_smoother()
        self.component.smooth(self.smoother)
        self.update_everything()


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
        main_layout = qtw.QVBoxLayout()
        self.setLayout(main_layout)

        if self._output_valid:
            main_layout.addWidget(qtw.QLabel("Mesh Output"))
            main_layout.addWidget(SettingsFileBox(
                self.component.settings, "mesh_output_path", system_path, "*.stl", "save", self.save_mesh
            ))
        if self._params_valid:
            self.component.settings.establish_defaults(
                parameters_path=str(pathlib.Path(system_path) / "parameters.dat")
            )
            main_layout.addWidget(qtw.QLabel("Parameters"))
            main_layout.addWidget(SettingsFileBox(
                self.component.settings, "parameters_path", system_path, "*.dat", "both",
                self.save_parameters, self.load_parameters
            ))
        if self._input_valid:
            main_layout.addWidget(qtw.QLabel("Mesh Input"))
            main_layout.addWidget(SettingsFileBox(
                self.component.settings, "mesh_input_path", system_path, "*.stl", "load", self.load_mesh
            ))
        if self._output_valid and self._params_valid:
            save_all_button = qtw.QPushButton("Save Everything")
            save_all_button.clicked.connect(self.save_all)
            main_layout.addWidget(save_all_button)

        # !!!!!!!!!!!!! this has to be moved to a generally re-parametrizable optic.
        """if self._params_valid:
            main_layout.addWidget(SettingsEntryBox(
                self.component.settings,
                "parameter_count",
                int,
                qtg.QIntValidator(5, 1000),
                [self.component.remesh, self.component.update]
            ))"""

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
                    ), outFile, pickle.HIGHEST_PROTOCOL)
                print(f"saved parameters for {self.component.name}: {self.component.settings.parameters_path}")
            except Exception:
                print(f"Exception while trying to save parameters")
                print(traceback.format_exc())

    def load_parameters(self):
        if self._params_valid:
            try:
                with open(self.component.settings.parameters_path, 'rb') as inFile:
                    params = pickle.load(inFile)
                    self.component.param_assign(params)
                print(f"loaded parameters for {self.component.name}: {self.component.settings.parameters_path}")
                self.driver.update_optics()
                self.driver.try_auto_redraw()
                if self.driver is not None:
                    self.driver.parameters_pane.refresh_parameters()
            except Exception:
                print(f"Exception while trying to load parameters")
                print(traceback.format_exc())

    def load_mesh(self):
        if self._input_valid:
            self.component.load(self.component.settings.mesh_input_path)
            self.driver.update_optics()
            self.driver.try_auto_redraw()
            try:
                self.component.drawer.draw()
            except Exception:
                pass
            print(f"loaded mesh for {self.component.name}: {self.component.settings.mesh_input_path}")

    def save_all(self):
        # These functions already check whether their operation is valid, and silently do nothing if it isn't
        self.save_mesh()
        self.save_parameters()


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
        self.setLayout(main_layout)

        # Set up the vum origin
        if self.component.enable_vum:
            self.component.settings.establish_defaults(vum_origin=[0.0, 0.0, 0.0])
            self.vum_origin_mesh = pv.PolyData(np.array(self.component.settings.vum_origin))
            self.vum_origin_actor = self.client.plot.add_mesh(
                self.vum_origin_mesh, render_points_as_spheres=True, color="red", point_size=15.0
            )

            main_layout.addWidget(
                SettingsVectorBox(
                    self.component,
                    "VUM Origin",
                    "vum_origin",
                    [self.update_vum_origin_display, self.component.try_mesh_tools]
                ), 0, 0, 1, 2
            )
            self.component.settings.establish_defaults(vum_origin_visible=False)
            main_layout.addWidget(
                SettingsCheckBox(
                    self.component, "Show vum origin", "vum_origin_visible", self.toggle_vum_origin_display
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
                SettingsVectorBox(
                    self.component,
                    "Acum Origin",
                    "accumulator_origin",
                    [self.update_acum_origin_display, self.component.try_mesh_tools]
                ), 1, 0, 1, 2
            )
            self.component.settings.establish_defaults(acum_origin_visible=False)
            main_layout.addWidget(
                SettingsCheckBox(
                    self.component, "Show accumulator origin", "acum_origin_visible", self.toggle_acum_origin_display
                ), 2, 0, 1, 1
            )
            self.toggle_acum_origin_display(self.component.settings.acum_origin_visible)

        # Set up the vertex update map
        if self.component.enable_vum:
            self.component.settings.establish_defaults(vum_active=True)
            self.component.settings.establish_defaults(vum_visible=False)
            main_layout.addWidget(
                SettingsCheckBox(self.component, "VUM Active", "vum_active"),
                3, 1, 1, 1
            )
            main_layout.addWidget(
                SettingsCheckBox(self.component, "VUM Visible", "vum_visible", self.update_vum_display),
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
                SettingsCheckBox(self.component, "Accumulator Active", "accumulator_active"),
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

            for face, vu in zip(self.component.faces.numpy(), self.component.vum.numpy()):
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

