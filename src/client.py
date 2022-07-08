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

import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import pyvistaqt as pvqt

import tfrt.engine as engine
import tfrt.operation as operation

import tfrt2.src.drawing as drawing
import tfrt2.src.settings as settings
import tfrt2.src.component_widgets as component_widgets
import tfrt2.src.wavelength as wavelength
import tfrt2.src.optics as optics


class OpticClientWindow(qtw.QWidget):
    def __init__(self):
        super().__init__(windowTitle="Linear Mark 5 Dev Client")
        self.control_pane_width = 350

        # Initialize the settings persistent data class
        self.settings_path = str(pathlib.Path(".") / "settings.dat")
        self.settings = settings.Settings()
        self.settings.establish_defaults(system_path=None)
        try:
            self.settings.load(self.settings_path)
        except Exception:
            print("Client: exception while trying to load the client settings:")
            print(traceback.format_exc())

        # Initialize all manner of base class variables
        self.ui = types.SimpleNamespace()
        self.plot = pvqt.QtInteractor()
        self.plot.camera.clipping_range = (.00001, 10000)
        self.plot.add_axes()
        self.display_pane = DisplayControls(self)
        self.trace_pane = TraceControls(self)
        self.parameters_pane = ParameterControls(self)
        self.components_pane = ComponentControls(self)
        self.pane_stack = None
        self._quit = threading.Event()
        self.retrace_button = qtw.QPushButton("Re-trace")
        self.retrace_button.setEnabled(False)

        # Build the UI
        self.build_ui([
            ("Display Settings", self.display_pane),
            ("Components", self.components_pane),
            ("Optimization", qtw.QWidget()),
            ("Tracing Controls", self.trace_pane),
            ("Parameters", self.parameters_pane),
        ])

        # Build the trace engine
        self.trace_engine = engine.OpticalEngine(
            3,
            [operation.StandardReaction()],
            simple_ray_inheritance={"wavelength"}
        )

        # Spawn a thread for performing local tracing
        self._update_rays_sig = UpdateRaysSignal()
        self._update_rays_sig.sig.connect(self._update_ray_drawer)

        self._start_local_trace = threading.Event()
        self._start_local_trace.clear()
        self._local_trace_thread = threading.Thread(
            target=self._local_trace_loop, daemon=True
        )
        self._local_trace_thread.start()

        # Try loading the most recently loaded system
        if self.settings.system_path is not None:
            self.load_system(None, path=self.settings.system_path)

        self.redraw()
        self.click_reset()

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
        reload_button.setEnabled(False)
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
        self.retrace_button.clicked.connect(self.retrace)
        redraw_layout.addWidget(self.retrace_button)
        clear_rays_button = qtw.QPushButton("Clear Rays")
        clear_rays_button.clicked.connect(self.trace_pane.clear_rays)
        redraw_layout.addWidget(clear_rays_button)
        reset_button = qtw.QPushButton("Reset Camera")
        reset_button.clicked.connect(self.click_reset)
        redraw_layout.addWidget(reset_button)
        update_toggle = qtw.QCheckBox("Always update system on redraw")
        update_toggle.setTristate(False)
        update_toggle.stateChanged.connect(self.click_update_toggle)
        self.settings.establish_defaults(auto_update_on_redraw=2)
        update_toggle.setCheckState(self.settings.auto_update_on_redraw)
        control_layout.addWidget(update_toggle)

        # A selector to choose which set of interface controls are visible
        tab_selector = qtw.QComboBox()
        tab_selector.insertItems(0, [pane[0] for pane in control_panes])
        tab_selector.currentIndexChanged.connect(self.change_active_pane)
        control_layout.addWidget(tab_selector)

        # Add the interface control stack
        self.pane_stack = qtw.QStackedWidget()
        for pane in control_panes:
            scroll_area = qtw.QScrollArea()
            scroll_area.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
            scroll_area.setWidgetResizable(True)
            pane[1].parent_scroll_area = scroll_area
            scroll_area.setWidget(pane[1])
            self.pane_stack.addWidget(scroll_area)
        # Try setting the selector via settings
        try:
            tab_selector.setCurrentIndex(self.settings.active_pane)
        except Exception:
            pass
        self.pane_stack.setMaximumWidth(self.control_pane_width)
        control_layout.addWidget(self.pane_stack)

        # Add the main 3D plot to the UI
        main_layout.addWidget(self.plot, 0, 1)

    def change_active_pane(self, i):
        self.pane_stack.setCurrentIndex(i)
        self.settings.active_pane = i

    def load_system(self, _, path=None):
        # If path was not specified, open a file dialog to select it.
        if path is None:
            try:
                parent_path = str(pathlib.Path(self.settings.system_path))
            except Exception:
                parent_path = str(pathlib.Path("."))
            dialog = qtw.QFileDialog(directory=parent_path)
            dialog.setFileMode(qtw.QFileDialog.Directory)
            if dialog.exec_():
                selected_files = dialog.selectedFiles()
                path = selected_files[0]

        # save the system settings now before overwriting or doing anything else
        self.save_system()

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
            self.trace_engine.optical_system = system_module.get_system(self)

            # If we reach here than we have achieved success!
            self.ui.loaded_label.setText(f"Current System: {str(path)}")
            self.ui.reload_button.setEnabled(True)
            self.retrace_button.setEnabled(True)
        except Exception:
            print(f"Client: got exception while trying to load the system at {path}:")
            print(traceback.format_exc())
            self.ui.loaded_label.setText("Current System: None")
            self.trace_engine.optical_system = None
            self.ui.reload_button.setEnabled(False)
            self.retrace_button.setEnabled(False)

        # Update the control panes with the new system
        self.populate_panes_from_system(self.trace_engine.optical_system)

    def populate_panes_from_system(self, system):
        self.display_pane.update_with_system(system)
        self.parameters_pane.update_with_system(system)
        self.components_pane.update_with_system(system)

        # Clear any traced rays
        self.trace_pane.ray_drawer.rays = None
        self.trace_pane.ray_drawer.draw()

    def redraw(self):
        if bool(self.settings.auto_update_on_redraw):
            if self.trace_engine.optical_system is not None:
                self.trace_engine.optical_system.update()
        self.display_pane.redraw()

    def retrace(self):
        self.retrace_button.setEnabled(False)
        self.retrace_button.setText("Working...")
        self._start_local_trace.set()

    def _local_trace_loop(self):
        while not self._quit.is_set():
            self._start_local_trace.wait()
            self._start_local_trace.clear()

            try:
                self.trace_engine.optical_system.update()
                self.trace_engine.ray_trace(self.settings.trace_depth)
            except AttributeError:
                # can fail if no system is present
                # print(traceback.format_exc())
                pass

            self._update_rays_sig.sig.emit()

    def _update_ray_drawer(self):
        self.display_pane.redraw()
        self.trace_pane.redraw()
        self.retrace_button.setEnabled(True)
        self.retrace_button.setText("Re-trace")

    def click_reset(self):
        self.plot.renderer.reset_camera()
        self.plot.camera.clipping_range = (.00001, 10000)

    def click_update_toggle(self, state):
        self.settings.auto_update_on_redraw = int(state)

    def reload_system(self):
        self.load_system(None, self.settings.system_path)

    def save_system(self):
        if self.trace_engine.optical_system is not None:
            try:
                self.trace_engine.optical_system.save()
            except Exception:
                print(f"Client: got exception while trying to save the current optical system:")
                print(traceback.format_exc())

    def quit(self):
        self._quit.set()
        try:
            self.settings.save(self.settings_path)
        except Exception:
            print(f"Client: failed to save system settings.")
        self.save_system()


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
        self.parent_client.pane_stack.updateGeometry()

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

        save_system_button = qtw.QPushButton("Save System")
        save_system_button.clicked.connect(self.save_system)
        self.main_layout.addWidget(save_system_button)

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

        self.updateGeometry()
        if self.parent_scroll_area:
            self.parent_scroll_area.updateGeometry()
        self.parent_client.pane_stack.updateGeometry()

    def save_system(self):
        for widget in self._added_widgets:
            try:
                widget.save_all()
            except Exception:
                pass


class TraceControls(qtw.QWidget):
    rayset_enum = {
        0: "None",
        1: "active_rays",
        2: "finished_rays",
        3: "dead_rays",
        4: "stopped_rays",
        5: "all_rays",
        6: "sources"
    }

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
        main_layout.addWidget(qtw.QLabel("Choose which rayset to display."), 2, 0, 3, 2)

        rayset_selector = qtw.QComboBox()
        rayset_selector.insertItems(0, [value.replace("_", " ") for value in self.rayset_enum.values()])
        rayset_selector.setCurrentIndex(parent.settings.active_set)

        def change_rayset(i):
            parent.settings.active_set = i

        change_rayset(parent.settings.active_set)
        rayset_selector.currentIndexChanged.connect(change_rayset)
        main_layout.addWidget(rayset_selector, 5, 0, 6, 2)

    def redraw(self):
        rayset = self.rayset_enum[self.parent_client.settings.active_set]
        if rayset == "None":
            self.ray_drawer.rays = None
        elif rayset == "sources":
            self.ray_drawer.rays = self.parent_client.trace_engine.optical_system.get_source_rays()
        else:
            self.ray_drawer.rays = getattr(self.parent_client.trace_engine, rayset)
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
                        controller = component_widgets.ParameterController(part)
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
    app = make_app()
    win = OpticClientWindow()
    main(app, win)
