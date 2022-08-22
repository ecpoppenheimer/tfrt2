"""
Whenever adding a new kind of file that has to be FTP'd, need to add the settings key to
ClientTCPWidget.keys_that_require_ftp (make sure the key has path in its name).  And then need to edit
tracing_TCP_server.reload_file to accept this kind of file.
"""

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
    # Connection state enum
    NO_CONNECTION = 0
    CONNECTED = 1
    PENDING = 2
    DISCONNECTED_LOCAL = 3
    DISCONNECTED_REMOTE = 4
    VALIDATION_FAIL = 5

    # Processing state enum
    UNKNOWN = 0
    PROCESSING = 1
    ABORTING = 2
    READY = 3

    client_settings_purges = {
        "system_path", "active_pane", "server_port", "server_ip", "auto_update_on_redraw", "auto_retrace"
    }
    component_settings_purges = {
        "visible", "color", "show_edges", "vum_origin_visible", "acum_origin_visible", "norm_arrow_visibility",
        "norm_arrow_length", "parameter_arrow_visibility", "parameter_arrow_length", "opacity", "show_spectrum",
        "vum_visible"
    }
    keys_that_require_ftp = {"mesh_input_path", "spectrum_path", "goal_image_path"}

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
        self.waiting_for_server_reset = False
        self.waiting_to_send_system = False

        # A lookup table of methods to call to process the data contained in various messages
        self.message_LUT = {
            tcp.SERVER_FULL: self.server_full,
            tcp.SERVER_ERROR: self.server_error,
            tcp.SERVER_M_SET_ACK: self.server_ack_driver_settings,
            tcp.SERVER_S_SET_ACK: self.server_ack_sys_settings,
            tcp.SERVER_FTP_ACK: self.server_ftp_ack,
            tcp.SERVER_SYS_L_ACK: self.server_system_load_ack,
            tcp.SERVER_TRACE_RSLT: self.client.remote_pane.receive_ray_trace_results,
            tcp.SERVER_PARAMS: self.receive_parameters,
            tcp.SERVER_PARAM_ACK: self.server_params_ack,
            tcp.SERVER_MESSAGE: self.print_server_message,
            tcp.SERVER_ILUM: self.client.remote_pane.receive_illuminance,
            tcp.SERVER_SYS_RST_ACK: self.server_ack_sys_reset,
            tcp.SERVER_READY: self.receive_server_ready,
            tcp.SERVER_BUSY: self.receive_server_busy,
            tcp.SERVER_FLATTENER: self.feed_flattener,
            tcp.SERVER_SINGLE_STEP: self.client.optimize_pane.receive_single_step,
            tcp.SERVER_ST_UPDATE: self.receive_status_update,
        }

        # Establish settings defaults
        self.client.settings.establish_defaults(server_port=tcp.DEFAULT_PORT, server_ip="localhost")

        # Build the UI
        layout = qtw.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
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
        ui_row = 0

        sync_layout.addWidget(qtw.QLabel("Synchronization with Server"), ui_row, 0, 1, 3)

        # A toggleable task pane showing the name of each task that has yet to be completed.
        self.tasks_visible_checkbox = qtw.QCheckBox("Show synchronization tasks")
        self.tasks_visible_checkbox.setCheckState(False)
        self.tasks_visible_checkbox.setTristate(False)
        self.tasks_visible_checkbox.clicked.connect(self.toggle_task_pane)
        sync_layout.addWidget(self.tasks_visible_checkbox, ui_row, 3, 1, 3)
        ui_row += 1

        self.task_pane = qtw.QWidget()
        self.task_pane.hide()
        self.task_layout = qtw.QVBoxLayout()
        self.task_pane.setLayout(self.task_layout)
        self.task_pane.setStyleSheet("background-color: white;")
        sync_layout.addWidget(self.task_pane, ui_row, 0, 1, 6)
        ui_row += 1

        # Button that will start the synchronization process
        self.sync_button = qtw.QPushButton("Synchronize")
        sync_layout.addWidget(self.sync_button, ui_row, 0, 1, 3)
        self.sync_button.clicked.connect(self.sync_with_server)

        # Button that will cancel the synchronization process
        self.abort_button = qtw.QPushButton("Abort Synchronization")
        self.abort_button.setEnabled(False)
        sync_layout.addWidget(self.abort_button, ui_row, 3, 1, 3)
        self.abort_button.clicked.connect(self.abort_sync)
        ui_row += 1

        # Progress bar indicating how many synchronization tasks remain to be completed
        self.progress_bar = qtw.QProgressBar()
        self.progress_bar.setRange(0, 1)
        sync_layout.addWidget(self.progress_bar, ui_row, 0, 1, 6)
        ui_row += 1

        # label indicating whether the sync was a success
        self.status_label = qtw.QLabel("Status: No connection.")
        sync_layout.addWidget(self.status_label, ui_row, 0, 1, 6)
        ui_row += 1

        # Display whether the server is busy with a long-running operation, and provide the ability to terminate
        # it, if so.
        sync_layout.addWidget(qtw.QLabel("Server:"), ui_row, 0, 1, 2)
        self.processing_indicator = qtw.QLabel("INIT")
        self.processing_indicator.setStyleSheet("border: 1px solid black;")
        sync_layout.addWidget(self.processing_indicator, ui_row, 2, 1, 2)
        self.processing_abort_button = qtw.QPushButton("Abort")
        self.processing_abort_button.setEnabled(False)
        self.processing_abort_button.clicked.connect(self.try_abort_processing)
        sync_layout.addWidget(self.processing_abort_button, ui_row, 4, 1, 2)
        self.set_processing_state(self.UNKNOWN)

        self.set_connection_state(self.NO_CONNECTION)

    def reset_system(self):
        self.abort_sync()
        self.set_status("System reset.")
        self.files_sent_to_server.clear()

        if self.server_socket is not None:
            self.server_socket.write(tcp.CLIENT_RESET_SYS)
            self.waiting_for_server_reset = True

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
        self.client.remote_pane.deactivate()
        self.client.optimize_pane.deactivate()
        self.set_processing_state(self.UNKNOWN)
        self.set_status("Disconnected.")

    def got_validation(self, success):
        if success:
            self.set_connection_state(self.CONNECTED)
            self.server_validated = True
            self.sync_controls.show()
            self.client.remote_pane.try_activate(self.server_socket)
            self.client.optimize_pane.try_activate(self.server_socket)
            self.set_status("Connection successful.")
        else:
            self.set_connection_state(self.VALIDATION_FAIL)
            self.server_validated = False
            self.sync_controls.hide()
            self.client.remote_pane.deactivate()
            self.client.optimize_pane.deactivate()
            self.set_status("Connection failed.")

    def socket_error(self, error):
        print(f"received socket error: {error}")
        self.close_connection(self.DISCONNECTED_REMOTE)
        self.server_validated = False

    def close_connection(self, new_state):
        self.sync_controls.hide()
        self.clean_sync_stuff()
        self.files_sent_to_server.clear()
        self.server_validated = False

        if self.server_socket is not None:
            self.set_connection_state(new_state)
            self.server_socket.close()
            self.server_socket = None

    def data_received(self, header, data):
        try:
            self.message_LUT[header](data)
        except KeyError:
            try:
                self.client.remote_pane.message_LUT[header](data)
            except KeyError:
                print(
                    "Got a message from server that could not be accepted by either the TCP widget nor the "
                    f"trace_controller: {str(header)}"
                )
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

    @staticmethod
    def server_error(data):
        context, message = pickle.loads(data)
        print("Nonfatal Exception raised on server: " + context)
        print(message)
        print("--End traceback--")

    def sync_with_server(self):
        if self.waiting_for_server_reset:
            self.waiting_to_send_system = True
            return
        self.waiting_to_send_system = False

        if self.client.optical_system is not None and self.server_socket.validated:
            self.clean_sync_stuff()
            self.abort_button.setEnabled(True)
            self.set_status("Sync pending.")

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
        self.set_status("Sync Aborted")

    def clean_sync_stuff(self):
        self.total_sync_tasks = 0
        self.completed_sync_tasks = 0
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

        self.set_status("Sync pending.", self.completed_sync_tasks, self.total_sync_tasks)

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
        self.set_status("Sync pending.", self.completed_sync_tasks, self.total_sync_tasks)
        if status:
            label.deleteLater()
            self.sync_tasks.pop(context)

            if context == "system_load":
                self.set_status("Sync success.")
            else:
                # If we have exactly one task left, 'system_load', then it is time to sent the system load request.
                # Don't do this if we received an error, because this could cause an infinite loop.
                if len(self.sync_tasks) == 1:
                    if tuple(self.sync_tasks.keys())[0] == "system_load":
                        self.server_socket.write(tcp.CLIENT_LOAD_SYSTEM)
                    else:
                        print("SyncError: ended up with just one sync_task, but it isn't system_load.")
        else:
            self.set_status("Sync failure.")
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

    def receive_parameters(self, data):
        for key, value in pickle.loads(data).items():
            self.client.optical_system.parts[key].param_assign(value)

    def send_parameters(self):
        all_params = {}
        for key, value in self.client.optical_system.parts.items():
            if hasattr(value, "parameters"):
                all_params[key] = value.parameters.numpy()
        self.server_socket.write(tcp.CLIENT_PARAMS, pickle.dumps(all_params))

    @staticmethod
    def print_server_message(data):
        print("Message received from server")
        print(pickle.loads(data))

    def server_ack_sys_reset(self, _):
        self.waiting_for_server_reset = False
        if self.waiting_to_send_system:
            self.sync_with_server()

    def try_abort_processing(self):
        self.set_processing_state(self.ABORTING)
        self.server_socket.write(tcp.CLIENT_ABORT_PROCS)
        self.client.optimize_pane.set_continuous_state(state=False)

    def set_processing_state(self, state):
        if state == self.PROCESSING:
            self.processing_indicator.setText("  processing")
            self.processing_indicator.setStyleSheet("QLabel { background-color: yellow}")
            self.processing_abort_button.setEnabled(True)
            self.set_state_frozen()
        elif state == self.READY:
            self.set_status("Completed.")
            self.processing_indicator.setText("  ready")
            self.processing_indicator.setStyleSheet("QLabel { background-color: green}")
            self.processing_abort_button.setEnabled(False)
            self.set_state_ready()
        elif state == self.ABORTING:
            self.processing_indicator.setText("  aborting")
            self.processing_indicator.setStyleSheet("QLabel { background-color: red}")
            self.processing_abort_button.setEnabled(False)
        else:
            self.processing_indicator.setText("  unknown")
            self.processing_indicator.setStyleSheet("QLabel { background-color: gray}")
            self.processing_abort_button.setEnabled(False)

    def set_state_frozen(self):
        self.sync_button.setEnabled(False)

    def set_state_ready(self):
        self.sync_button.setEnabled(True)

    def receive_server_ready(self, _):
        self.set_processing_state(self.READY)

    def receive_server_busy(self, _):
        self.set_processing_state(self.PROCESSING)

    def feed_flattener(self, data):
        flattening_density = pickle.loads(data)
        try:
            flattener = self.client.optical_system.goal.flattening_icdf
            flattener.clear_density()
            flattener.accumulate_density(flattening_density)
            flattener.compute(direction="inverse")
        except AttributeError:
            pass

    def set_status(self, text, current=None, high=None):
        if current is not None:
            if high is None:
                high = current
        else:
            current = 1
            high = 1
        self.progress_bar.setRange(0, high)
        self.progress_bar.setValue(current)

        self.status_label.setText("Status: " + text)

    def receive_status_update(self, data):
        self.set_status(*pickle.loads(data))

