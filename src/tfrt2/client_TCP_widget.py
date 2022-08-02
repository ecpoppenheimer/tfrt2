from pathlib import Path
import pickle
import traceback

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtNetwork as qtn

import tfrt2.tcp_base as tcp_base
import tfrt2.component_widgets as cw
import tfrt2.tcp_base as tcp


class ClientTCPWidget(qtw.QWidget):
    """
    Please note that in order to use this widget inside a large application, the application must call reactor.run()
    before instantiating this widget to get the twisted message loop running.
    """
    NO_CONNECTION = 0
    CONNECTED = 1
    PENDING = 2
    DISCONNECTED_LOCAL = 3
    DISCONNECTED_REMOTE = 4
    VALIDATION_FAIL = 5

    client_settings_purges = {
        "system_path", "active_pane", "server_port", "server_ip", "auto_update_on_redraw", "auto_retrace"
    }
    component_settings_purges = {
        "visible", "color", "show_edges", "vum_origin_visible", "acum_origin_visible", "norm_arrow_visibility",
        "norm_arrow_length", "parameter_arrow_visibility", "parameter_arrow_length", "opacity", "show_spectrum",
        "vum_visible"
    }
    keys_that_require_ftp = {"mesh_input_path", "spectrum_path"}

    def __init__(self, client):
        super().__init__()
        self.client = client
        self.connection_state = None
        self.server_socket = None
        self.server_validated = False
        self.sync_tasks = {}
        self.files_sent_to_server = {}
        self.total_sync_tasks = 0
        self.completed_sync_tasks = 0

        # A lookup table of methods to call to process the data contained in various messages
        self.message_LUT = {
            tcp.SERVER_FULL: self.server_full,
            tcp.SERVER_ERROR: self.server_error,
            tcp.SERVER_M_SET_ACK: self.server_ack_driver_settings,
            tcp.SERVER_S_SET_ACK: self.server_ack_sys_settings,
            tcp.SERVER_FTP_ACK: self.server_ftp_ack,
            tcp.SERVER_SYS_L_ACK: self.server_system_load_ack,
            tcp.SERVER_TRACE_RSLT: self.receive_ray_trace_results,
            tcp.SERVER_PARAMS: self.receive_parameters,
            tcp.SERVER_PARAM_ACK: self.server_params_ack,
        }

        # Establish settings defaults
        self.client.settings.establish_defaults(server_port=tcp.DEFAULT_PORT, server_ip="localhost")

        # Build the UI
        layout = qtw.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel("TCP Connection to server"), 0, 0, 1, 2)

        # Main button for connection / disconnection
        self.connection_button = qtw.QPushButton("Connect")
        self.connection_button.clicked.connect(self.click_connection_button)
        layout.addWidget(self.connection_button, 1, 0)

        # Port
        self.port_widget = cw.SettingsEntryBox(
            self.client.settings, "server_port", int, validator=qtg.QIntValidator(0, 65535)
        )
        layout.addWidget(self.port_widget, 1, 1)

        # IP
        self.ip_widget = cw.SettingsEntryBox(self.client.settings, "server_ip", str, validator=tcp.IP4Validator())
        layout.addWidget(self.ip_widget, 2, 1)
        self.ip_widget.edit_box.textChanged.connect(self.ip_widget_text_changed)

        # Status indicator
        self.status_indicator = qtw.QLabel("")
        self.status_indicator.setFixedWidth(150)
        layout.addWidget(self.status_indicator, 2, 0)

        # Toggleable widget containing controls that only make sense when a valid connection is established
        self.sync_controls = qtw.QWidget()
        layout.addWidget(self.sync_controls, 3, 0, 1, 2)
        self.sync_controls.hide()
        sync_layout = qtw.QGridLayout()
        self.sync_controls.setLayout(sync_layout)

        sync_layout.addWidget(qtw.QLabel("Synchronization with Server"), 0, 0, 1, 2)

        # Button that will start the synchronization process
        sync_button = qtw.QPushButton("Synchronize")
        sync_layout.addWidget(sync_button, 1, 0)
        sync_button.clicked.connect(self.sync_with_server)

        # Button that will cancel the synchronization process
        self.abort_button = qtw.QPushButton("Abort Synchronization")
        self.abort_button.setEnabled(False)
        sync_layout.addWidget(self.abort_button, 1, 1)
        self.abort_button.clicked.connect(self.abort_sync)

        # Progress bar indicating how many synchronization tasks remain to be completed
        self.sync_progress = qtw.QProgressBar()
        self.sync_progress.setRange(0, 1)
        sync_layout.addWidget(self.sync_progress, 2, 0, 1, 2)

        # label indicating whether the sync was a success
        self.sync_status = qtw.QLabel("Status: ")
        sync_layout.addWidget(self.sync_status, 3, 1)

        # A toggleable task pane showing the name of each task that has yet to be completed.
        self.tasks_visible_checkbox = qtw.QCheckBox("Show synchronization tasks")
        self.tasks_visible_checkbox.setCheckState(False)
        self.tasks_visible_checkbox.setTristate(False)
        self.tasks_visible_checkbox.clicked.connect(self.toggle_task_pane)
        sync_layout.addWidget(self.tasks_visible_checkbox, 3, 0)

        self.task_pane = qtw.QWidget()
        self.task_pane.hide()
        self.task_layout = qtw.QVBoxLayout()
        self.task_pane.setLayout(self.task_layout)
        self.task_pane.setStyleSheet("background-color: white;")
        sync_layout.addWidget(self.task_pane, 4, 0, 1, 2)

        # A button to obtain a ray trace from the server
        self.trace_button = qtw.QPushButton("Remote Raytrace")
        self.trace_button.setEnabled(False)
        self.trace_button.clicked.connect(self.request_ray_trace)
        sync_layout.addWidget(self.trace_button, 5, 0)

        self.set_connection_state(self.NO_CONNECTION)

    def reset_system(self):
        self.abort_sync()
        self.files_sent_to_server.clear()
        self.trace_button.setEnabled(False)

    def click_connection_button(self):
        if self.connection_state == self.CONNECTED:
            # Disconnect
            self.close_connection(self.DISCONNECTED_LOCAL)
        elif self.connection_state == self.PENDING:
            # Cancel
            self.close_connection(self.NO_CONNECTION)
        else:
            # Start a new connection.
            self.set_connection_state(self.PENDING)
            self.connect(self.client.settings.server_ip, self.client.settings.server_port)

    def connect(self, address, port):
        self.server_socket = qtn.QTcpSocket(self)
        self.server_socket.connectToHost(address, port)
        self.server_socket.connected.connect(self.got_connection)

    def got_connection(self):
        self.server_socket = tcp_base.TcpDataPacker(
            self.server_socket,
            tcp_base.CLIENT_ACK,
            tcp_base.SERVER_ACK
        )
        self.server_socket.disconnected.connect(self.disconnected)
        self.server_socket.errorOccurred.connect(self.socket_error)
        self.server_socket.data_ready.sig.connect(self.data_received)
        self.server_socket.connection_validated.sig.connect(self.got_validation)

    def disconnected(self):
        self.close_connection(self.DISCONNECTED_REMOTE)

    def got_validation(self, success):
        if success:
            self.set_connection_state(self.CONNECTED)
            self.server_validated = True
            self.sync_controls.show()
        else:
            self.set_connection_state(self.VALIDATION_FAIL)
            self.server_validated = False
            self.sync_controls.hide()

    def socket_error(self, error):
        print(f"received socket error: {error}")
        self.close_connection(self.DISCONNECTED_REMOTE)
        self.server_validated = False

    def close_connection(self, new_state):
        self.sync_controls.hide()
        self.clean_sync_stuff()
        self.files_sent_to_server.clear()
        self.trace_button.setEnabled(False)
        self.server_validated = False

        if self.server_socket is not None:
            self.set_connection_state(new_state)
            self.server_socket.close()
            self.server_socket = None

    def data_received(self, header, data):
        try:
            self.message_LUT[header](data)
        except Exception:
            print("hit exception in data_received...")
            print(traceback.format_exc())

    def ip_widget_text_changed(self):
        state = self.ip_widget.edit_box.validator().validate(self.ip_widget.edit_box.text(), 0)[0]
        if state == qtg.QValidator.Acceptable:
            color = "white"
        else:
            color = "pink"
        self.ip_widget.setStyleSheet("QLineEdit { background-color: %s }" % color)

    def set_connection_state(self, state):
        self.connection_state = state
        if state == self.NO_CONNECTION:
            self.status_indicator.setText("No Connection.")
            self.connection_button.setText("Connect")
        elif state == self.PENDING:
            self.status_indicator.setText("Attempting to connect...")
            self.connection_button.setText("Cancel")
        elif state == self.CONNECTED:
            self.status_indicator.setText("Connection successful!")
            self.connection_button.setText("Disconnect")
        elif state == self.DISCONNECTED_LOCAL:
            self.status_indicator.setText("Disconnected.")
            self.connection_button.setText("Connect")
        elif state == self.DISCONNECTED_REMOTE:
            self.status_indicator.setText("Lost connection.")
            self.connection_button.setText("Connect")
        elif state == self.VALIDATION_FAIL:
            self.status_indicator.setText("Server handshake failed.")
            self.connection_button.setText("Connect")
        else:
            self.status_indicator.setText("Unknown error - wrong state!")
            self.connection_button.setText("Connect")

    def server_full(self, data):
        self.close_connection(self.DISCONNECTED_REMOTE)
        self.status_indicator.setText("Server already has connection!")

    def server_error(self, data):
        context, message = pickle.loads(data)
        print("Nonfatal Exception raised on server: " + context)
        print(message)
        print("--End traceback--")

    def sync_with_server(self):
        if self.client.optical_system is not None and self.server_socket.validated:
            self.clean_sync_stuff()
            self.abort_button.setEnabled(True)
            self.trace_button.setEnabled(False)
            self.sync_status.setText("Status: Pending")

            self.add_sync_task("driver_settings", self.get_clean_driver_settings())
            self.add_sync_task("system_settings", self.get_clean_system_settings())
            self.add_sync_task("send_parameters")
            self.add_sync_task("system_load")

            # Send the script.  This command checks to make sure it is needed (changed or not already sent).
            self.ftp("optical_script.py", Path(self.client.settings.system_path) / "optical_script.py")

            # Do not want to send the load system command until we have received acknowledgement that every other
            # task has been completed.  The actual load command will be sent by _server_ack, once the load task is
            # the only remaining task.

    def abort_sync(self):
        self.clean_sync_stuff()
        self.sync_status.setText("Status: Aborted")

    def clean_sync_stuff(self):
        self.total_sync_tasks = 0
        self.sync_progress.setMaximum(1)
        self.completed_sync_tasks = 0
        self.sync_progress.setValue(0)
        self.abort_button.setEnabled(False)

        for v in self.sync_tasks.values():
            v.deleteLater()
        self.sync_tasks.clear()

    def add_sync_task(self, mode, *args):
        if mode == "driver_settings":
            title, key = "Transfer driver settings", "driver_settings"
            settings = args[0]
            self.server_socket.write(tcp.CLIENT_MAIN_SET, pickle.dumps(settings))
        elif mode == "system_settings":
            title, key = "Transfer system settings", "system_settings"
            settings = args[0]
            self.server_socket.write(tcp.CLIENT_SYS_SET, pickle.dumps(settings))
        elif mode == "system_load":
            title, key = "Load system", "system_load"
            # Don't actually send this command, let _server_ack handle that once all other commands have been
            # acknowledged.
        elif mode == "send_parameters":
            title, key = "Transfer Parameters", "send_parameters"
            self.send_parameters()
        elif mode == "file_transfer":
            local_filename = Path(args[1])
            title, key = "FTP " + str(local_filename.stem) + str(local_filename.suffix), args[0]
        else:
            raise ValueError(f"ClientTCPWidget: invalid mode {mode} given to add_sync_task")

        label = qtw.QLabel(title)
        self.sync_tasks[key] = label
        self.task_layout.addWidget(label)

        self.total_sync_tasks += 1
        self.sync_progress.setMaximum(self.total_sync_tasks)

    def get_clean_driver_settings(self):
        return {k: v for k, v in self.client.settings.dict.items() if k not in self.client_settings_purges}

    def _server_ack(self, status, context):
        try:
            label = self.sync_tasks[context]
        except KeyError:
            print(
                f"Unexpected signal from server!  Acknowledged set {context} but was not asked to.  Was the syc "
                "aborted?"
            )
            return

        self.completed_sync_tasks += 1
        self.sync_progress.setValue(self.completed_sync_tasks)
        if status:
            label.deleteLater()
            self.sync_tasks.pop(context)

            if context == "system_load":
                self.sync_status.setText("Status: Success")
            else:
                # If we have exactly one task left, 'system_load', then it is time to sent the system load request.
                # Don't do this if we received an error, because this could cause an infinite loop.
                if len(self.sync_tasks) == 1:
                    if tuple(self.sync_tasks.keys())[0] == "system_load":
                        self.server_socket.write(tcp.CLIENT_LOAD_SYSTEM)
                    else:
                        print("SyncError: ended up with just one sync_task, but it isn't system_load.")
        else:
            self.sync_status.setText("Status: Failure")
            label.setStyleSheet("background-color: pink;")



    def server_ack_driver_settings(self, data):
        self._server_ack(pickle.loads(data), "driver_settings")

    def server_ack_sys_settings(self, data):
        self._server_ack(pickle.loads(data), "system_settings")

    def server_ftp_ack(self, data):
        self._server_ack(*pickle.loads(data))

    def server_params_ack(self, data):
        self._server_ack(pickle.loads(data), "send_parameters")

    def server_system_load_ack(self, data):
        self._server_ack(pickle.loads(data), "system_load")
        self.trace_button.setEnabled(True)
        self.abort_button.setEnabled(False)

    def get_clean_system_settings(self):
        clean_settings = {}
        for name in self.client.optical_system.parts.keys():
            component_settings = {}
            for key, value in self.client.optical_system.settings.dict[name].dict.items():
                if key in self.keys_that_require_ftp:
                    # name is the name of the component (guaranteed to be unique),
                    # key is the settings key that points to this file, guaranteed to be unique within the component
                    # thus naming the remote filename with their combination is guaranteed to create a unique identifier
                    # that can be used to keep track of the transferred file on both ends.  This file should be FTP'd
                    # over to the server, and the new unique filename should be fed to the settings that will be
                    # sent over.
                    if '|' in name or '|' in key or '$' in name or '$' in key:
                        raise ValueError(
                            "ClientTCPWidget: '|' and '$' are invalid characters to include in names / paths"
                        )
                    remote_filename = name + '|' + key.replace("_path", "") + '|' + str(Path(value).suffix)
                    print(f"sending an FTP with filename {remote_filename}")
                    self.ftp(remote_filename, value)
                    component_settings[key] = "$PATH$" + remote_filename
                elif key not in self.component_settings_purges and "path" not in key:
                    component_settings[key] = value
                # The if / elif can ignore some cases: settings keys with path in their name but not in
                # keys_that_require_ftp, but this is by design - we don't need to bother transferring some keys, like
                # those for mesh outputs or parameters - these will never be used by the server.
            clean_settings[name] = component_settings
        return clean_settings

    def ftp(self, remote_filename, local_path):
        current_mtime = Path(local_path).stat().st_mtime

        if remote_filename in self.files_sent_to_server.keys():
            cached_mtime, cached_full_path = self.files_sent_to_server[remote_filename]
            # This file has already been sent
            if cached_full_path == local_path and current_mtime <= cached_mtime:
                # This file has not changed, so nothing needs to be done
                return

        # If we reach here, the file either has not been sent yet or has been updated since it was last sent, so send
        # it now.
        self.files_sent_to_server[remote_filename] = (current_mtime, local_path)
        self.add_sync_task("file_transfer", remote_filename, local_path)

        self.server_socket.write(tcp.CLIENT_FTP, pickle.dumps((remote_filename, Path(local_path).read_bytes())))

    def toggle_task_pane(self):
        if self.tasks_visible_checkbox.checkState():
            self.task_pane.show()
        else:
            self.task_pane.hide()

    def quit(self):
        self.close_connection(self.DISCONNECTED_LOCAL)
        self.server_validated = False

    def check_system_state(self):
        if self.client.optical_system is not None and self.server_validated:
            self.sync_controls.show()
        else:
            self.sync_controls.hide()

    def receive_ray_trace_results(self, data):
        self.client.trace_pane.ray_drawer.rays = pickle.loads(data)
        self.client.trace_pane.ray_drawer.draw()

    def request_ray_trace(self):
        self.server_socket.write(tcp.CLIENT_RQST_TRACE)

    def receive_parameters(self, data):
        for key, value in pickle.loads(data).items():
            self.client.optical_system.parts[key].param_assign(value)

    def send_parameters(self):
        all_params = {}
        for key, value in self.client.optical_system.parts.items():
            if hasattr(value, "parameters"):
                all_params[key] = value.parameters.numpy()
        self.server_socket.write(tcp.CLIENT_PARAMS, pickle.dumps(all_params))
