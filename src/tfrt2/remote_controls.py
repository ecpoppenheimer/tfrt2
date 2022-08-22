import pickle

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg

import tfrt2.component_widgets as cw
import tfrt2.tcp_base as tcp


class RemotePane(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()

        self.server_socket = None
        self.parent_client = parent
        self._active = False

        # Build the base UI
        base_layout = qtw.QVBoxLayout()

        self.error_widget = qtw.QLabel("Connect to a trace server to activate remote controls.")
        base_layout.addWidget(self.error_widget)
        self.active_widget = qtw.QWidget()
        self.active_widget.hide()
        base_layout.addWidget(self.active_widget)

        base_layout.addStretch()
        self.setLayout(base_layout)

        # Build the controls in the active widget
        main_layout = qtw.QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setColumnMinimumWidth(0, 11)
        main_layout.setColumnStretch(0, 0)
        self.ui_row = 0
        self.active_widget.setLayout(main_layout)

        # Remote raytrace
        illuminance_label = qtw.QLabel(
            "Perform a remote ray trace.  Scales rays from each source by ray count factor.  Ray count factor is also "
            "used by remote tracing operations to scale the batch size.  This should be as high as possible without "
            "causing out of memory errors."
        )
        illuminance_label.setWordWrap(True)
        main_layout.addWidget(illuminance_label, self.ui_row, 0, 1, 7)
        self.ui_row += 1

        self.parent_client.settings.establish_defaults(ray_count_factor=1.0)
        self.remote_raytrace_button = qtw.QPushButton("Remote Raytrace")
        self.remote_raytrace_button.clicked.connect(self.remote_raytrace)
        main_layout.addWidget(self.remote_raytrace_button, self.ui_row, 1, 1, 3)

        self.ray_count_factor_box = cw.SettingsEntryBox(
            self.parent_client.settings, "ray_count_factor", float, qtg.QDoubleValidator(1e-4, 1e6, 6),
            callback=self.parent_client.feed_ray_count_factor
        )
        main_layout.addWidget(self.ray_count_factor_box, self.ui_row, 4, 1, 3)
        self.ui_row += 1

        # Controls for tracing to make the illuminance plot (also defines the goal flatten icdf)
        illuminance_label = qtw.QLabel(
            "Trace the system to produce an illuminance plot.  This will also feed the goal's flattener.  These "
            "settings can over-ride the settings on the components page / goal, but will only override for this "
            "operation, not subsequent automatic goal flattener updates done during optimization."
        )
        illuminance_label.setWordWrap(True)
        main_layout.addWidget(illuminance_label, self.ui_row, 0, 1, 7)

        self.ui_row += 1
        self.illuminance_button = qtw.QPushButton("Calculate Illuminance")
        self.illuminance_button.clicked.connect(self.request_illuminance_plot)
        main_layout.addWidget(self.illuminance_button, self.ui_row, 1, 1, 3)
        self.ui_row += 1

        self.parent_client.settings.establish_defaults(
            rq_illum_override=True, rq_illum_ray_count=int(1e6), rq_illum_x_res=64, rq_illum_y_res=64,
            rq_illum_x_min=-1.0, rq_illum_x_max=1.0, rq_illum_y_min=-1.0, rq_illum_y_max=1.0
        )

        main_layout.addWidget(cw.SettingsCheckBox(
            self.parent_client.settings, "rq_illum_override", "Over-ride goal settings"
        ), self.ui_row, 4, 1, 3)
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "rq_illum_ray_count", int, qtg.QIntValidator(1000, int(1e9)),
            label="Minimum Rays"
        ), self.ui_row, 1, 1, 3)
        self.ui_row += 1

        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "rq_illum_x_res", int, qtg.QIntValidator(4, 4096),
            label="X res"
        ), self.ui_row, 1, 1, 3)
        main_layout.addWidget(cw.SettingsEntryBox(
            self.parent_client.settings, "rq_illum_y_res", int, qtg.QIntValidator(4, 4096),
            label="Y res"
        ), self.ui_row, 4, 1, 3)
        self.ui_row += 1

        self.rq_illum_x_lims = cw.SettingsRangeBox(
            self.parent_client.settings, "X extents", "rq_illum_x_min", "rq_illum_x_max", float,
            qtg.QDoubleValidator(-1e6, 1e6, 9),
        )
        main_layout.addWidget(self.rq_illum_x_lims, self.ui_row, 1, 1, 4)
        self.rq_illum_y_lims = cw.SettingsRangeBox(
            self.parent_client.settings, "Y extents", "rq_illum_y_min", "rq_illum_y_max", float,
            qtg.QDoubleValidator(-1e6, 1e6, 9),
        )
        main_layout.addWidget(self.rq_illum_y_lims, self.ui_row+1, 1, 1, 4)
        self.rq_illum_pull_lims_button = qtw.QPushButton("Pull from goal")
        self.rq_illum_pull_lims_button.clicked.connect(self.pull_rq_illum_lims)
        main_layout.addWidget(self.rq_illum_pull_lims_button, self.ui_row, 5, 2, 2)
        self.ui_row += 2

    def try_activate(self, socket=None):
        if socket is not None:
            self.server_socket = socket
        if self.parent_client.optical_system is None:
            self.error_widget.setText("Load an optical system to activate.")
            self._deactivate()
        elif self.parent_client.optical_system.goal is None:
            self.error_widget.setText("Optical system requires a goal to activate remote controls.")
            self._deactivate()
        elif self.server_socket is None:
            self.error_widget.setText("Connect to a trace server to activate remote controls.")
            self._deactivate()
        else:
            # are able to activate
            self._activate()

    def _activate(self):
        self._active = True
        self.error_widget.hide()
        self.active_widget.show()

    def _deactivate(self):
        self._active = True
        self.error_widget.show()
        self.active_widget.hide()

    def deactivate(self):
        self.active_widget.hide()
        self.server_socket = None
        self.error_widget.setText("Connect to a trace server to activate remote controls.")
        self.error_widget.show()
        self._active = False

    def request_illuminance_plot(self):
        if self._active:
            if self.parent_client.settings.rq_illum_override:
                ilum_settings = (
                    self.parent_client.settings.rq_illum_ray_count, self.parent_client.settings.ray_count_factor,
                    self.parent_client.settings.rq_illum_x_res, self.parent_client.settings.rq_illum_y_res,
                    self.parent_client.settings.rq_illum_x_min, self.parent_client.settings.rq_illum_x_max,
                    self.parent_client.settings.rq_illum_y_min, self.parent_client.settings.rq_illum_y_max
                )
            else:
                ilum_settings = (self.parent_client.settings.ray_count_factor,)
            self.parent_client.tcp_widget.set_status("Measuring Illuminance.", 0, 1)
            self.server_socket.write(tcp.CLIENT_RQST_ILUM, pickle.dumps(ilum_settings))

    def receive_illuminance(self, data):
        illuminance = pickle.loads(data)
        if self.parent_client.optical_system.goal is not None:
            self.parent_client.optical_system.goal.feed_flatten(illuminance)
        self.parent_client.ui.illuminance_widget.set_data(illuminance, self.get_extents())
        self.parent_client.draw_illuminance_box(self.get_extents(), self.get_goal_box())
        self.parent_client.ui.illuminance_widget.draw()

    def get_extents(self):
        plot_x_min = self.parent_client.settings.rq_illum_x_min
        plot_x_max = self.parent_client.settings.rq_illum_x_max
        plot_y_min = self.parent_client.settings.rq_illum_y_min
        plot_y_max = self.parent_client.settings.rq_illum_y_max
        return plot_x_min, plot_x_max, plot_y_min, plot_y_max

    def get_goal_box(self):
        goal_x_min = self.parent_client.optical_system.goal.settings.c1_min
        goal_x_max = self.parent_client.optical_system.goal.settings.c1_max
        goal_y_min = self.parent_client.optical_system.goal.settings.c2_min
        goal_y_max = self.parent_client.optical_system.goal.settings.c2_max

        return goal_x_min, goal_x_max, goal_y_min, goal_y_max

    def pull_rq_illum_lims(self):
        self.rq_illum_x_lims.set_range(
            self.parent_client.optical_system.settings.goal.c1_min,
            self.parent_client.optical_system.settings.goal.c1_max
        )
        self.rq_illum_y_lims.set_range(
            self.parent_client.optical_system.settings.goal.c2_min,
            self.parent_client.optical_system.settings.goal.c2_max
        )

    def remote_raytrace(self):
        self.parent_client.tcp_widget.set_status("Remote raytrace...", 0, 1)
        self.parent_client.trace_pane.clear_rays()
        self.server_socket.write(tcp.CLIENT_RQST_TRACE, pickle.dumps(self.parent_client.settings.ray_count_factor))

    def receive_ray_trace_results(self, data):
        self.parent_client.optical_system.feed_raysets(pickle.loads(data))
        self.parent_client.trace_pane.redraw()
