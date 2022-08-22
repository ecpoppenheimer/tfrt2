# Disable redundant TF warnings
if __name__ == "__main__":
    import logging
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.FATAL)

import importlib.util
import pathlib
import types
import sys
import traceback
import threading
import signal

import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import pyvistaqt as pvqt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import tfrt2.drawing as drawing
import tfrt2.settings as settings
import tfrt2.component_widgets as component_widgets
import tfrt2.wavelength as wavelength
import tfrt2.optics as optics
import tfrt2.client_TCP_widget as tcp_widget
from tfrt2.remote_controls import RemotePane
from tfrt2.optimize import OptimizationPane


class OpticClientWindow(qtw.QWidget):
    driver_type = "client"
    rayset_enum = {
        0: "None",
        1: "active_rays",
        2: "finished_rays",
        3: "dead_rays",
        4: "stopped_rays",
        5: "all_rays",
        6: "source_rays",
        7: "unfinished_rays"
    }

    def __init__(self):
        super().__init__(windowTitle="Linear Mark 5 Dev Client")
        self.control_pane_width = 350

        # Initialize the settings persistent data class
        self.settings_path = str(pathlib.Path(__file__).parent / "settings.dat")
        self.settings = settings.Settings()
        self.settings.establish_defaults(system_path=None)
        try:
            self.settings.load(self.settings_path)
        except Exception:
            print("Client: exception while trying to load the client settings:")
            print(traceback.format_exc())

        # 3D system plot
        self.plot = pvqt.QtInteractor()
        self.plot.camera.clipping_range = (.00001, 10000)
        self.plot.add_axes()

        # Initialize all manner of base class variables
        self.ui = types.SimpleNamespace()
        self.optical_system = None
        self.display_pane = DisplayControls(self)
        self.trace_pane = TraceControls(self)
        self.parameters_pane = ParameterControls(self)
        self.components_pane = ComponentControls(self)
        self.optimize_pane = OptimizationPane(self)
        self.remote_pane = RemotePane(self)
        self.ui.pane_stack = None
        self._quit = threading.Event()
        self.ui.retrace_button = qtw.QPushButton("Re-trace")
        self.ui.retrace_button.setEnabled(False)
        self.tcp_widget = tcp_widget.ClientTCPWidget(self)

        # Illuminance plot
        plt.style.use("dark_background")
        self.ui.illuminance_widget = component_widgets.MplImshowWidget(blank=True, alignment=(.05, .05, .9, .9))
        self.ui.illuminance_widget.box = None

        # Spawn a thread for performing local tracing
        self._update_rays_sig = UpdateRaysSignal()
        self._update_rays_sig.sig.connect(self._update_ray_drawer)

        self._start_local_trace = threading.Event()
        self._start_local_trace.clear()
        self._local_trace_thread = threading.Thread(
            target=self._local_trace_loop, daemon=True
        )
        self._local_trace_thread.start()

        # Build the UI
        self.build_ui([
            ("Display Settings", self.display_pane),
            ("Components", self.components_pane),
            ("Remote Operations", self.remote_pane),
            ("Optimization Controls", self.optimize_pane),
            ("Tracing Controls", self.trace_pane),
            ("Parameters", self.parameters_pane),
        ])

        # Try loading the most recently loaded system
        if self.settings.system_path is not None:
            self.load_system(None, path=self.settings.system_path)

    def build_ui(self, control_panes):
        main_layout = qtw.QGridLayout()
        self.setLayout(main_layout)

        # Setup the control layout
        control_layout = qtw.QVBoxLayout()
        control_widget = qtw.QWidget()
        control_widget.setMaximumWidth(self.control_pane_width)
        control_widget.setLayout(control_layout)
        main_layout.addWidget(control_widget, 0, 0)

        # Add a button and label to load an optical system
        load_layout = qtw.QHBoxLayout()
        control_layout.addLayout(load_layout)
        load_button = qtw.QPushButton("Load")
        load_button.clicked.connect(self.load_system)
        load_button.setMaximumWidth(35)
        load_layout.addWidget(load_button)
        reload_button = qtw.QPushButton("Reload")
        reload_button.clicked.connect(self.reload_system)
        reload_button.setEnabled(True)
        reload_button.setMaximumWidth(45)
        self.ui.reload_button = reload_button
        load_layout.addWidget(reload_button)
        self.ui.loaded_label = qtw.QLabel("Current System: None")
        self.ui.loaded_label.setAlignment(qtc.Qt.AlignRight)

        load_layout.addWidget(self.ui.loaded_label)

        # A button to redraw the 3D plot, and also to reset the camera
        redraw_layout = qtw.QHBoxLayout()
        control_layout.addLayout(redraw_layout)
        redraw_button = qtw.QPushButton("Redraw")
        redraw_button.clicked.connect(self.redraw)
        redraw_layout.addWidget(redraw_button)
        self.ui.retrace_button.clicked.connect(self.retrace)
        redraw_layout.addWidget(self.ui.retrace_button)
        clear_rays_button = qtw.QPushButton("Clear Rays")
        clear_rays_button.clicked.connect(self.trace_pane.clear_rays)
        redraw_layout.addWidget(clear_rays_button)
        reset_button = qtw.QPushButton("Reset Camera")
        reset_button.clicked.connect(self.click_reset)
        redraw_layout.addWidget(reset_button)

        # Check boxes to control automatic updating
        update_layout = qtw.QHBoxLayout()
        control_layout.addLayout(update_layout)
        update_toggle = qtw.QCheckBox("Update system on redraw")
        update_toggle.setTristate(False)
        update_toggle.stateChanged.connect(self.click_update_toggle)
        self.settings.establish_defaults(auto_update_on_redraw=2)
        update_toggle.setCheckState(self.settings.auto_update_on_redraw)
        update_layout.addWidget(update_toggle)

        retrace_toggle = qtw.QCheckBox("Retrace")
        retrace_toggle.setTristate(False)
        retrace_toggle.stateChanged.connect(self.toggle_auto_retrace)
        self.settings.establish_defaults(auto_retrace=2)
        retrace_toggle.setCheckState(self.settings.auto_retrace)
        update_layout.addWidget(retrace_toggle)

        # The TCP client widget
        control_layout.addWidget(self.tcp_widget)

        # A selector to choose which set of interface controls are visible
        tab_selector = qtw.QComboBox()
        tab_selector.insertItems(0, [pane[0] for pane in control_panes])
        tab_selector.currentIndexChanged.connect(self.change_active_pane)
        control_layout.addWidget(tab_selector)

        # Add the interface control stack
        self.ui.pane_stack = qtw.QStackedWidget()
        for pane in control_panes:
            scroll_area = qtw.QScrollArea()
            scroll_area.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
            scroll_area.setWidgetResizable(True)
            pane[1].parent_scroll_area = scroll_area
            scroll_area.setWidget(pane[1])
            self.ui.pane_stack.addWidget(scroll_area)
        # Try setting the selector via settings
        try:
            tab_selector.setCurrentIndex(self.settings.active_pane)
        except Exception:
            pass
        self.ui.pane_stack.setMaximumWidth(self.control_pane_width)
        control_layout.addWidget(self.ui.pane_stack)

        # Add a tabbed area to show visualizations
        visualization_tabs = qtw.QTabWidget()
        main_layout.addWidget(visualization_tabs, 0, 1)

        # Add the main 3D plot to the UI
        visualization_tabs.addTab(self.plot, "3D System Visualization")

        # Add a 2D illuminance plot to the UI
        visualization_tabs.addTab(self.ui.illuminance_widget, "Illuminance Plot")

    def change_active_pane(self, i):
        self.ui.pane_stack.setCurrentIndex(i)
        self.settings.active_pane = i

    def load_system(self, _, path=None):
        # If path was not specified, open a file dialog to select it.
        if path is None:
            try:
                parent_path = str(pathlib.Path(self.settings.system_path))
            except Exception:
                parent_path = str(pathlib.Path(".."))
            dialog = qtw.QFileDialog(directory=parent_path)
            dialog.setFileMode(qtw.QFileDialog.Directory)
            if dialog.exec_():
                selected_files = dialog.selectedFiles()
                path = selected_files[0]

        if path is None:
            # Opened a file selection box above, but if the user closes it instead of selecting a file, will get none
            # here, in which case just do nothing
            return

        # save the system settings now before overwriting or doing anything else
        self.save_system()
        self.tcp_widget.reset_system()
        if self.optical_system is not None:
            self.optical_system.cleanup()

        # load the system at this location
        try:
            path = pathlib.Path(path)
            # Code found at https://stackoverflow.com/questions/27189044/import-with-dot-name-in-python
            # Unfortunately my dev environment is placed in a folder with a period in the name, and this cannot
            # be changed.
            spec = importlib.util.spec_from_file_location(
                name="system_module",
                location=str(path / "optical_script.py")
            )
            system_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(system_module)

            self.settings.system_path = path
            self.optical_system = system_module.get_system(self)
            self.optical_system.post_init()

            # If we reach here than we have achieved success!
            self.ui.loaded_label.setText(f"Current System: {str(path)}")
            self.ui.retrace_button.setEnabled(True)

            self.redraw()
            self.try_auto_retrace()
            self.click_reset()
        except FileNotFoundError:
            print("Invalid file")
            self.ui.loaded_label.setText("Current System: None")
            self.optical_system = None
            self.ui.retrace_button.setEnabled(False)
        except Exception:
            print(f"Client: got exception while trying to load the system at {path}:")
            print(traceback.format_exc())
            self.ui.loaded_label.setText("Current System: None")
            self.optical_system = None
            self.ui.retrace_button.setEnabled(False)

        # Update the control panes with the new system
        self.populate_panes_from_system(self.optical_system)
        self.tcp_widget.check_system_state()
        self.remote_pane.try_activate()
        self.optimize_pane.try_activate()

    def populate_panes_from_system(self, system):
        self.display_pane.update_with_system(system)
        self.parameters_pane.update_with_system(system)
        self.components_pane.update_with_system(system)

        # Clear any traced rays
        self.trace_pane.ray_drawer.rays = None
        self.trace_pane.ray_drawer.draw()

    def redraw(self):
        if bool(self.settings.auto_update_on_redraw):
            if self.optical_system is not None:
                self.optical_system.update()
        self.display_pane.redraw()
        self.trace_pane.redraw()

    def update_optics(self):
        if bool(self.settings.auto_update_on_redraw):
            if self.optical_system is not None:
                self.optical_system.update_optics()
        self.display_pane.redraw()

    def retrace(self):
        self.ui.retrace_button.setEnabled(False)
        self.ui.retrace_button.setText("Working...")
        self._start_local_trace.set()

    def try_auto_retrace(self):
        if bool(self.settings.auto_retrace) and self.optical_system is not None:
            self.retrace()

    def toggle_auto_retrace(self, state):
        self.settings.auto_retrace = int(state)

    def _local_trace_loop(self):
        while not self._quit.is_set():
            self._start_local_trace.wait()
            self._start_local_trace.clear()

            if self.optical_system is not None:
                try:
                    self.optical_system.update()
                    self.optical_system.ray_trace(self.settings.trace_depth)
                    #self.optical_system.fast_trace(self.settings.trace_depth)
                    #sample = self.optical_system.get_trace_samples(self.settings.trace_depth)
                    #self.optical_system.precompiled_trace(self.settings.trace_depth, sample)
                except Exception:
                    print(f"Client: got exception while tracing")
                    print(traceback.format_exc())
                finally:
                    self._update_rays_sig.sig.emit()

    def _update_ray_drawer(self):
        self.trace_pane.redraw()
        self.ui.retrace_button.setEnabled(True)
        self.ui.retrace_button.setText("Re-trace")

    def click_reset(self):
        self.plot.renderer.reset_camera()
        #self.plot.camera.clipping_range = (.00001, 10000)

    def click_update_toggle(self, state):
        self.settings.auto_update_on_redraw = int(state)

    def reload_system(self):
        self.load_system(None, self.settings.system_path)

    def save_system(self):
        if self.optical_system is not None:
            try:
                self.optical_system.save()
            except Exception:
                print(f"Client: got exception while trying to save the current optical system:")
                print(traceback.format_exc())

    def quit(self):
        self._quit.set()
        self.tcp_widget.quit()
        self.optimize_pane.history_widget.shut_down()
        try:
            self.settings.save(self.settings_path)
        except Exception:
            print(f"Client: failed to save system settings.")
        self.save_system()

    def feed_ray_count_factor(self):
        self.optimize_pane.ray_count_factor_box.edit_box.setText(str(self.settings.ray_count_factor))
        self.remote_pane.ray_count_factor_box.edit_box.setText(str(self.settings.ray_count_factor))

    def draw_illuminance_box(self, plot_extents, goal_box):
        p_x_min, p_x_max, p_y_min, p_y_max = plot_extents
        g_x_min, g_x_max, g_y_min, g_y_max = goal_box
        a_x_min = min(g_x_min, p_x_min)
        a_x_max = max(g_x_max, p_x_max)
        a_y_min = min(g_y_min, p_y_min)
        a_y_max = max(g_y_max, p_y_max)

        if self.ui.illuminance_widget.box is not None:
            self.ui.illuminance_widget.box.remove()
        self.ui.illuminance_widget.box = self.ui.illuminance_widget.ax.add_patch(mpl.patches.Rectangle(
            (g_x_min, g_y_min),
            g_x_max - g_x_min,
            g_y_max - g_y_min,
            edgecolor="yellow",
            facecolor="none",
            linewidth=1
        ))
        self.ui.illuminance_widget.ax.set_xlim(a_x_min, a_x_max)
        self.ui.illuminance_widget.ax.set_ylim(a_y_min, a_y_max)


class UpdateRaysSignal(qtc.QObject):
    sig = qtc.pyqtSignal()


class DisplayControls(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_client = parent
        self._added_widgets = []
        self.optical_controllers = []
        self.target_controllers = []
        self.stop_controllers = []

        self.main_layout = qtw.QVBoxLayout()
        self.setLayout(self.main_layout)
        main_label = qtw.QLabel(
            "These controls affect display only - they do not affect the tracing or optimization results."
        )
        main_label.setWordWrap(True)
        main_label.setMinimumWidth(self.parent_client.control_pane_width-40)
        self.main_layout.addWidget(main_label)
        self.main_layout.addStretch()

    def add_widget(self, widget):
        self.main_layout.insertWidget(self.main_layout.count() - 1, widget)
        self._added_widgets.append(widget)

    def update_with_system(self, system):
        for widget in self._added_widgets:
            if hasattr(widget, "remove_drawer"):
                widget.remove_drawer()
            widget.deleteLater()
        self._added_widgets = []
        self.optical_controllers = []
        self.target_controllers = []
        self.stop_controllers = []

        if system is not None:
            for part in system.opticals:
                controller = optics.TrigBoundaryDisplayController(part, self.parent_client.plot)
                self.add_widget(qtw.QLabel(part.name))
                self.add_widget(controller)
                self.optical_controllers.append(controller)

            for part in system.stops:
                controller = optics.TrigBoundaryDisplayController(part, self.parent_client.plot)
                self.add_widget(qtw.QLabel(part.name))
                self.add_widget(controller)
                self.stop_controllers.append(controller)

            for part in system.targets:
                controller = optics.TrigBoundaryDisplayController(part, self.parent_client.plot)
                self.add_widget(qtw.QLabel(part.name))
                self.add_widget(controller)
                self.target_controllers.append(controller)

        self.updateGeometry()
        if self.parent_scroll_area:
            self.parent_scroll_area.updateGeometry()
        self.parent_client.ui.pane_stack.updateGeometry()

        self.redraw()

    def redraw(self):
        for collection in (self.optical_controllers, self.target_controllers, self.stop_controllers):
            for controller in collection:
                controller.redraw()


class ComponentControls(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_client = parent
        self._added_widgets = []

        self.main_layout = qtw.QVBoxLayout()
        self.setLayout(self.main_layout)

        """save_system_button = qtw.QPushButton("Save System")
        save_system_button.clicked.connect(self.save_system)
        self.main_layout.addWidget(save_system_button)"""

        self.main_layout.addStretch()

    def add_widget(self, widget):
        self.main_layout.insertWidget(self.main_layout.count() - 1, widget)
        self._added_widgets.append(widget)

    def update_with_system(self, system):
        for widget in self._added_widgets:
            if hasattr(widget, "remove_drawer"):
                widget.remove_drawer()
            widget.deleteLater()
        self._added_widgets = []

        if system is not None:
            for part in system.parts.values():
                try:
                    controller_widgets = part.controller_widgets
                except AttributeError:
                    controller_widgets = []
                if controller_widgets:
                    self.add_widget(qtw.QLabel(part.name))
                    for widget in controller_widgets:
                        widget.client = self.parent_client
                        self.add_widget(widget)
            # Try adding the goal widget, if it exists
            if system.goal is not None:
                self.add_widget(qtw.QLabel("Goal"))
                self.add_widget(system.goal.controller)

        self.updateGeometry()
        if self.parent_scroll_area:
            self.parent_scroll_area.updateGeometry()
        self.parent_client.ui.pane_stack.updateGeometry()

    """def save_system(self):
        for widget in self._added_widgets:
            try:
                widget.save_all()
            except Exception:
                pass"""


class TraceControls(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_client = parent

        # Set up the ray drawer
        parent.settings.establish_defaults(
            active_set=4,
            trace_depth=3,
            min_wavelength=wavelength.VISIBLE_MIN,
            max_wavelength=wavelength.VISIBLE_MAX
        )
        self.ray_drawer = drawing.RayDrawer3D(
            parent.plot,
            min_wavelength=parent.settings.min_wavelength,
            max_wavelength=parent.settings.max_wavelength
        )

        # Make the UI for this controller
        stretch_layout = qtw.QVBoxLayout()
        self.setLayout(stretch_layout)
        stretch_widget = qtw.QWidget()
        main_layout = qtw.QGridLayout()
        stretch_widget.setLayout(main_layout)
        stretch_layout.addWidget(stretch_widget)
        stretch_layout.addStretch()

        # Control to adjust the trace depth
        trace_depth_explainer = qtw.QLabel("Trace depth affects both local display and remote optimization.")
        trace_depth_explainer.setWordWrap(True)
        trace_depth_explainer.setMinimumWidth(self.parent_client.control_pane_width - 40)
        main_layout.addWidget(trace_depth_explainer, 0, 0, 1, 2)

        trace_depth_label = qtw.QLabel("Trace Depth")
        main_layout.addWidget(trace_depth_label, 1, 0)
        trace_depth_line = qtw.QLineEdit(self)
        trace_depth_line.setText(str(parent.settings.trace_depth))
        trace_depth_line.setValidator(qtg.QIntValidator(1, 1000))

        def set_trace_depth():
            parent.settings.trace_depth = int(trace_depth_line.text())

        trace_depth_line.editingFinished.connect(set_trace_depth)
        main_layout.addWidget(trace_depth_line, 1, 1)

        # Control to select which rays are drawn
        main_layout.addWidget(qtw.QLabel("Choose which rayset to display."), 2, 0, 1, 2)

        rayset_selector = qtw.QComboBox()
        rayset_selector.insertItems(0, [value.replace("_", " ") for value in self.parent_client.rayset_enum.values()])
        rayset_selector.setCurrentIndex(parent.settings.active_set)

        def change_rayset(i):
            parent.settings.active_set = i
            self.redraw()

        change_rayset(parent.settings.active_set)
        rayset_selector.currentIndexChanged.connect(change_rayset)
        main_layout.addWidget(rayset_selector, 3, 0, 1, 2)

        # The commented out block was created as a debug tool, for the precompiled trace.  I am leaving the code here
        # though I hope I never need to use it again.
        """def click_get_samples():
            if parent.optical_system is not None:
                params, source_rays, triangles = parent.optical_system.get_trace_samples(parent.settings.trace_depth)
                print("parameters")
                for name, p in zip(parent.optical_system.parametric_optics, params):
                    print(f"{name}: {p.shape}")
                print(f"source rays: {source_rays.shape}")
                trig_map, trig_map_indices = triangles
                for i in range(len(trig_map_indices) - 1):
                    print(f"iteration {i} triangles: {trig_map[trig_map_indices[i]:trig_map_indices[i+1]]}")

        get_samples_button = qtw.QPushButton("Get Trace Samples")
        get_samples_button.clicked.connect(click_get_samples)
        main_layout.addWidget(get_samples_button, 4, 0)

        def click_trace_from_samples():
            if parent.optical_system is not None:
                sample = parent.optical_system.get_trace_samples(parent.settings.trace_depth)
                parent.optical_system.precompiled_trace(parent.settings.trace_depth, sample)
                self.redraw()

        trace_samples_button = qtw.QPushButton("Get Samples and Trace")
        trace_samples_button.clicked.connect(click_trace_from_samples)
        main_layout.addWidget(trace_samples_button, 4, 1)"""

    def redraw(self):
        if self.parent_client.optical_system is None:
            self.ray_drawer.rays = None
        else:
            rayset = self.parent_client.rayset_enum[self.parent_client.settings.active_set]
            if rayset == "None":
                self.ray_drawer.rays = None
            else:
                self.parent_client.optical_system.refresh_source_rays()
                self.ray_drawer.rays = getattr(self.parent_client.optical_system, rayset)
                try:
                    self.parent_client.optical_system.goal.try_redraw_goal()
                except Exception:
                    pass
        self.ray_drawer.draw()

    def clear_rays(self):
        self.ray_drawer.rays = None
        self.ray_drawer.draw()


class ParameterControls(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_client = parent
        self._added_widgets = []
        self.parameter_controllers = []

        self.main_layout = qtw.QVBoxLayout()
        self.setLayout(self.main_layout)
        main_label = qtw.QLabel(
            "Controls for the parameters of each parametric optic."
        )
        self.main_layout.addWidget(main_label)
        self.main_layout.addStretch()

    def add_widget(self, widget):
        self.main_layout.insertWidget(self.main_layout.count() - 1, widget)
        self._added_widgets.append(widget)

    def update_with_system(self, system):
        for widget in self._added_widgets:
            if hasattr(widget, "remove_drawer"):
                widget.remove_drawer()
            widget.deleteLater()
        self._added_widgets = []

        if system is not None:
            for part in system.opticals:
                if hasattr(part, "parameters"):
                        controller = component_widgets.ParameterController(part, self.parent_client)
                        self.add_widget(qtw.QLabel(part.name))
                        self.add_widget(controller)
                        self.parameter_controllers.append(controller)

    def refresh_parameters(self):
        for each in self.parameter_controllers:
            each.refresh_parameters()


def make_app():
    # Global exception hook to pick up on exceptions thrown in worker threads.
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    return qtw.QApplication([])


def main(app, window):
    app.aboutToQuit.connect(window.quit)
    try:
        window.showMaximized()
    finally:
        exit(app.exec_())


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = make_app()
    win = OpticClientWindow()
    main(app, win)
