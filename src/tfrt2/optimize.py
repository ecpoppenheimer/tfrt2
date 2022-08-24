import pickle
import threading
import time

import numpy as np
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg

import tfrt2.component_widgets as cw
import tfrt2.tcp_base as tcp


class OptimizationPane(qtw.QWidget):
    def __init__(self, parent_client):
        super().__init__()

        self.server_socket = None
        self.parent_client = parent_client
        self._active = False
        self.parent_client.settings.establish_defaults(
            op_ray_sample_size=100,
            op_smooth_period=0,
            op_learning_rate=1.0,
            op_momentum=0.0,
            op_grad_clip_low=0.0,
            op_grad_clip_high=0.4,
            op_illuminance_period=10,
        )
        self._running_continuous = False
        self.step_count = 0

        # Build the base UI
        base_layout = qtw.QVBoxLayout()

        self.error_widget = qtw.QLabel("Connect to a trace server to activate optimization controls.")
        base_layout.addWidget(self.error_widget)
        self.active_widget = qtw.QWidget()
        self.active_widget.hide()
        base_layout.addWidget(self.active_widget)

        base_layout.addStretch()
        self.setLayout(base_layout)

        # Build the controls in the active widget
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        ui_row = 0
        self.active_widget.setLayout(main_layout)

        # Ray count factor and minimum rays
        self.parent_client.settings.establish_defaults(
            ray_count_factor=1.0,
            op_step_ray_count=1000,
            op_update_on_step_result=True,
        )
        self.ray_count_factor_box = cw.SettingsEntryBox(
            self.parent_client.settings, "ray_count_factor", float, qtg.QDoubleValidator(1e-4, 1e6, 6),
            callback=self.parent_client.feed_ray_count_factor
        )
        main_layout.addWidget(self.ray_count_factor_box, ui_row, 0, 1, 3)
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_step_ray_count", int, qtg.QIntValidator(0, int(1e9)),
            label="Minimum rays/step."
        ), ui_row, 3, 1, 3)
        ui_row += 1

        # Display step count and run reset
        label = qtw.QLabel("Resetting the run counter will also delete all parameter history.")
        label.setWordWrap(True)
        main_layout.addWidget(label, ui_row, 0, 1, 6)
        ui_row += 1

        self.step_label = qtw.QLabel("Step: 0")
        main_layout.addWidget(self.step_label, ui_row, 0, 1, 2)

        reset_run_button = qtw.QPushButton("Reset run")
        reset_run_button.clicked.connect(self.click_reset_run)
        main_layout.addWidget(reset_run_button, ui_row, 2, 1, 2)

        # Toggle switch to update the display when a result is received
        main_layout.addWidget(cw.SettingsCheckBox(
            self.parent_client.settings, "op_update_on_step_result", "Update Display"
        ), ui_row, 4, 1, 2)
        ui_row += 1

        separator = qtw.QFrame()
        separator.setFrameShape(qtw.QFrame.Shape.HLine)
        main_layout.addWidget(separator, ui_row, 0, 1, 6)
        ui_row += 1

        # The parameter history widget
        self.history_widget = ParameterHistoryWidget(parent_client)
        main_layout.addWidget(self.history_widget, ui_row, 0, 1, 6)
        ui_row += 1

        separator = qtw.QFrame()
        separator.setFrameShape(qtw.QFrame.Shape.HLine)
        main_layout.addWidget(separator, ui_row, 0, 1, 6)
        ui_row += 1

        # Display sample size
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_ray_sample_size", int, qtg.QIntValidator(0, int(1e6)),
            label="Display Sample Size"
        ), ui_row, 0, 1, 4)
        ui_row += 1

        # Learning rate and momentum
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_learning_rate", float, qtg.QDoubleValidator(0, 1e4, 10),
            label="Learning rate"
        ), ui_row, 0, 1, 3)
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_momentum", float, qtg.QDoubleValidator(0, 1.0, 4),
            label="Momentum"
        ), ui_row, 3, 1, 3)
        ui_row += 1

        # Grad clip
        label = qtw.QLabel(
            "Gradient clipping is relative to the value of each individual parameter.  Set low to zero to disable "
            "minimum magnitude clipping, which forces the optic to change every step and is only recommended near "
            "the beginning of the run."
        )
        label.setWordWrap(True)
        main_layout.addWidget(label, ui_row, 0, 1, 6)
        ui_row += 1
        main_layout.addWidget(cw.SettingsRangeBox(
            self.parent_client.settings, "Gradient Clip", "op_grad_clip_low", "op_grad_clip_high", float,
            qtg.QDoubleValidator(0, 10.0, 6),
        ), ui_row, 0, 1, 6)
        ui_row += 1

        main_layout.addWidget(qtw.QLabel("Gradient statistics from optimizer:"), ui_row, 0, 1, 6)
        ui_row += 1
        grad_stats_pane = qtw.QWidget()
        self.grad_stats_layout = qtw.QVBoxLayout()
        grad_stats_pane.setLayout(self.grad_stats_layout)
        self.grad_stats_units = {}
        main_layout.addWidget(grad_stats_pane, ui_row, 0, 1, 6)
        ui_row += 1

        # Smooth period
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_smooth_period", int, qtg.QIntValidator(0, 100),
            label="Smoother Period (0 to disable smoothing)."
        ), ui_row, 0, 1, 6)
        ui_row += 1

        # Illuminance Period
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "op_illuminance_period", int, qtg.QIntValidator(0, 100),
            label="Illuminance Period (0 to disable)."
        ), ui_row, 0, 1, 6)
        ui_row += 1

        # Button to launch a single step
        self.single_step_button = qtw.QPushButton("Single Step")
        self.single_step_button.clicked.connect(self.single_step)
        main_layout.addWidget(self.single_step_button, ui_row, 0, 1, 3)

        # Button to continuously run single steps
        self.continuous_step_button = qtw.QPushButton("Continuous Run")
        self.continuous_step_button.clicked.connect(self.set_continuous_state)
        main_layout.addWidget(self.continuous_step_button, ui_row, 3, 1, 3)
        ui_row += 1

        # Button to calculate illuminance, so I don't have to switch tabs
        illuminance_button = qtw.QPushButton("Calculate Illuminance")
        illuminance_button.clicked.connect(self.parent_client.remote_pane.request_illuminance_plot)
        main_layout.addWidget(illuminance_button, ui_row, 0, 1, 3)

    def try_activate(self, socket=None):
        if socket is not None:
            self.server_socket = socket
        if self.parent_client.optical_system is None:
            self.deactivate("Load an optical system to optimize.")
        elif self.parent_client.optical_system.goal is None:
            self.deactivate("Optical system requires a goal to optimize.")
        elif self.server_socket is None:
            self.deactivate("Connect to a trace server to optimize.")
        else:
            # are able to activate
            self._activate()
            self.history_widget.enable_playback(False)
            self.history_widget.reset_history()

    def _activate(self):
        self._active = True
        self.error_widget.hide()
        self.active_widget.show()

        # Populate the grad stats
        for component in self.parent_client.optical_system.parametric_optics:
            label = qtw.QLabel(f"{component.name}. mean: ---, variance: ---")
            self.grad_stats_units[component.name] = label
            self.grad_stats_layout.addWidget(label)

    def deactivate(self, text="Connect to a trace server to optimize."):
        self.active_widget.hide()
        self.server_socket = None
        self.error_widget.setText(text)
        self.error_widget.show()
        self._active = False
        self.history_widget.enable_playback(False)

        # Remove the grad stats
        for label in self.grad_stats_units.values():
            label.deleteLater()
        self.grad_stats_units = {}

    def single_step(self):
        # Check to see if we need to calculate the illuminance first
        if not self.parent_client.optical_system.goal.flatten_ready:
            self.parent_client.remote_pane.request_illuminance_plot()

        self.parent_client.tcp_widget.set_status("Single Optimization Step.", 0, 1)
        if self.parent_client.settings.op_smooth_period > 0:
            do_smooth = self.step_count % self.parent_client.settings.op_smooth_period == 0
        else:
            do_smooth = False
        optimize_params = (
            do_smooth,
            self.parent_client.settings.op_learning_rate,
            self.parent_client.settings.op_momentum,
            self.parent_client.settings.op_grad_clip_low,
            self.parent_client.settings.op_grad_clip_high,
        )
        self.server_socket.write(
            tcp.CLIENT_SINGLE_STEP,
            pickle.dumps((
                self.parent_client.settings.op_step_ray_count,
                self.parent_client.settings.ray_count_factor,
                self.parent_client.settings.op_ray_sample_size,
                optimize_params
            ))
        )

    def receive_single_step(self, data):
        finished_rays, new_params, grad_stats = pickle.loads(data)

        # Update the display.  Updates both the mesh display and the rays.
        if self.parent_client.settings.op_update_on_step_result and not self.history_widget.override_redraw:
            for component, p in zip(self.parent_client.optical_system.parametric_optics, new_params):
                component.param_assign(p)
            self.parent_client.optical_system.update_optics()
            self.parent_client.optical_system.clear_rays()
            self.parent_client.optical_system.feed_raysets({"finished_rays": finished_rays, "all_rays": finished_rays})
            self.parent_client.display_pane.redraw()
            self.parent_client.trace_pane.redraw()

        # Update the grad stats
        for name, mean, variance in grad_stats:
            self.grad_stats_units[name].setText(f"{name}. mean: {mean:.4f}, variance: {variance:.4f}")

        # Possibly queue the next step
        self.history_widget.add_history(new_params)
        self.increment_step()
        if self._running_continuous:
            if self.step_count % self.parent_client.settings.op_illuminance_period == 0:
                self.parent_client.remote_pane.request_illuminance_plot()

            self.single_step()

    def set_continuous_state(self, _=None, state=None):
        if state is None:
            self._running_continuous = not self._running_continuous
        else:
            self._running_continuous = state

        self.single_step_button.setEnabled(not self._running_continuous)
        # self.routine_button.setEnabled(self._running_continuous)
        if self._running_continuous:
            self.single_step()
            self.continuous_step_button.setText("Stop")
        else:
            self.continuous_step_button.setText("Continuous Run")

    def click_reset_run(self):
        self.step_count = 0
        self.step_label.setText("Step: 0")
        self.history_widget.reset_history()

    def increment_step(self):
        self.step_count += 1
        self.step_label.setText(f"Step: {self.step_count}")


class ParameterHistoryWidget(qtw.QWidget):
    def __init__(self, parent_client):
        super().__init__()
        self.parent_client = parent_client
        self.parameter_history = []
        self.system = None
        self.override_redraw = False
        self._animation_frame_number = None

        self.parent_client.settings.establish_defaults(
            ph_fps=2,
            ph_start_frame=0,
            ph_end_frame=0,
        )

        # Thread stuff for the animation thread
        self._shutdown = threading.Event()
        self._shutdown.clear()
        self._playback = threading.Event()
        self._playback.clear()
        self._animate_thread = threading.Thread(target=self._animate_loop, daemon=True)
        self._animate_thread.start()

        # Build the UI
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        ui_row = 1
        self.setLayout(main_layout)
        main_layout.addWidget(qtw.QLabel("Parameter History"), 0, 0, 1, 6)

        # save and load buttons
        save_button = qtw.QPushButton("Save")
        save_button.clicked.connect(self.save)
        main_layout.addWidget(save_button, ui_row, 0, 1, 3)

        load_button = qtw.QPushButton("Load")
        load_button.clicked.connect(self.load)
        main_layout.addWidget(load_button, ui_row, 3, 1, 3)
        ui_row += 1

        # slice parameters
        label = qtw.QLabel(
            "Positive numbers are relative to the start of the list, negative relative to the end of the list, zero "
            "anchors to the end of the list."
        )
        label.setWordWrap(True)
        main_layout.addWidget(label, ui_row, 0, 1, 6)
        ui_row += 1

        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "ph_start_frame", int, qtg.QIntValidator(-10000, 10000), label="Start"
        ), ui_row, 0, 1, 2)
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "ph_end_frame", int, qtg.QIntValidator(-10000, 10000), label="End"
        ), ui_row, 2, 1, 2)

        main_layout.addWidget(qtw.QLabel("Current"), ui_row, 4, 1, 1)
        self.current_frame_box = qtw.QLineEdit("0")
        self.current_frame_box.setValidator(qtg.QIntValidator(0, 10000))
        self.current_frame_box.editingFinished.connect(self.set_current_frame)
        main_layout.addWidget(self.current_frame_box, ui_row, 5, 1, 1)
        ui_row += 1

        self.play_button = qtw.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        main_layout.addWidget(self.play_button, ui_row, 0, 1, 2)

        self.loop_check_box = qtw.QCheckBox("Loop")
        self.loop_check_box.setTristate(False)
        self.loop_check_box.setCheckState(2)
        main_layout.addWidget(self.loop_check_box, ui_row, 2, 1, 4)

        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "ph_fps", float, qtg.QDoubleValidator(.1, 100, 3), label="FPS"
        ), ui_row, 4, 1, 2)
        ui_row += 1

    def add_history(self, data):
        self.parameter_history.append(data)

    def reset_history(self):
        self.parameter_history = []

    def save(self):
        if self.parent_client.optical_system is None:
            print("Cannot save, no optical system is present")
            return
        selected_file, _ = qtw.QFileDialog.getSaveFileName(
            directory=str(self.parent_client.settings.system_path), filter="*.dat"
        )
        if selected_file:
            with open(selected_file, 'wb') as outFile:
                pickle.dump((
                    self.get_system_spec(),
                    self.parameter_history
                ), outFile)

    def load(self):
        if self.parent_client.optical_system is None:
            print("Cannot load, no optical system is present")
            return
        selected_file, _ = qtw.QFileDialog.getOpenFileName(
            directory=str(self.parent_client.settings.system_path), filter="*.dat"
        )
        if selected_file:
            with open(selected_file, 'rb') as inFile:
                remote_spec, p_history = pickle.load(inFile)

            # validate the system spec
            local_spec = self.get_system_spec()
            for remote, local in zip(remote_spec, local_spec):
                if remote != local:
                    print(
                        "Loading history failed, because the saved parameters do not match those of the current system"
                    )
                    return
            self.parameter_history = p_history
            self.parent_client.optimize_pane.step_count = len(self.parameter_history)
            self.parent_client.optimize_pane.step_label.setText(f"Step: {self.parent_client.optimize_pane.step_count}")

    def get_system_spec(self):
        return tuple(
            (component.name, component.parameters.shape[0])
            for component in self.parent_client.optical_system.parametric_optics
        )

    def shut_down(self):
        self._shutdown.set()

    def enable_playback(self, state):
        if state:
            self.override_redraw = True
            self._playback.set()
            self.play_button.setText("Stop")
        else:
            self.override_redraw = False
            self._playback.clear()
            self.play_button.setText("Play")

    def toggle_play(self):
        if self._playback.is_set():
            self.enable_playback(False)
        else:
            self._animation_frame_number = None
            self.enable_playback(True)

    def reset(self):
        self.enable_playback(False)
        self._animation_frame_number = None

    def _animate_loop(self):
        while not self._shutdown.is_set():
            self._playback.wait()
            self._animate_frame()
            if self.parent_client.settings.ph_fps == 0:
                delay = 1
            else:
                delay = 1/self.parent_client.settings.ph_fps
            time.sleep(delay)

    def _animate_frame(self):
        # Parse the animation frame number.  May be invalid, or may be negative, in which case it is relative to
        # the end of the list.  Either way, ensure it is relative to the start of the list, and non-negative
        if len(self.parameter_history) == 0:
            print("No parameter history to animate")
            self.enable_playback(False)
            return

        if self._animation_frame_number is None:
            self._animation_frame_number = self.parent_client.settings.ph_start_frame
            if self._animation_frame_number < 0:
                self._animation_frame_number += len(self.parameter_history)
            self._animation_frame_number = np.clip(self._animation_frame_number, 0, len(self.parameter_history) - 1)

        # Do the current frame
        self._show_frame(self._animation_frame_number)

        # Check whether we have reached the end, in which case we need to either stop or loop.
        end_frame = self.parent_client.settings.ph_end_frame
        if end_frame < 0:
            end_frame += len(self.parameter_history)
        end_frame = np.clip(end_frame, 0, len(self.parameter_history) - 1)
        if end_frame == 0:
            end_frame = len(self.parameter_history) - 1

        self._animation_frame_number += 1
        if self._animation_frame_number > end_frame:
            # At the end of the animation
            self._animation_frame_number = None
            if not self.loop_check_box.checkState():
                self.enable_playback(False)

    def set_current_frame(self):
        current = int(self.current_frame_box.text())
        if current < len(self.parameter_history) - 1:
            self._animation_frame_number = current
            self.current_frame_box.setStyleSheet("QLineEdit { background-color: white}")
            self._show_frame(self._animation_frame_number)
        else:
            self.current_frame_box.setStyleSheet("QLineEdit { background-color: pink}")

    def _show_frame(self, frame):
        self.current_frame_box.setText(str(frame))
        params = self.parameter_history[frame]
        for component, p in zip(self.parent_client.optical_system.parametric_optics, params):
            component.param_assign(p)
        self.parent_client.optical_system.update_optics()
        self.parent_client.display_pane.redraw()
