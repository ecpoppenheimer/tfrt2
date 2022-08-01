# Disable redundant TF warnings
import tcp_base

if __name__ == "__main__":
    import logging
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.FATAL)

import sys
from pathlib import Path
import pickle
import importlib.util
import traceback
import signal

import numpy as np
import PyQt5.QtNetwork as qtn
import PyQt5.QtWidgets as qtw

import tfrt2.src.tcp_base as tcp
import tfrt2.src.settings as settings


class TraceServer(qtn.QTcpServer):
    driver_type = "server"
    temp_path = Path(__file__).parent.resolve() / "server_temp"

    def __init__(self, port):
        super().__init__()
        self.optical_system = None
        self.listen(port=port)
        self.newConnection.connect(self.got_connection)
        self.client_socket = None
        self.settings = settings.Settings(system_path=str(self.temp_path))
        self.cached_component_settings = None
        self.cached_parameters = None
        self.received_files = {}

        self.clean_temp()

        self.message_LUT = {
            tcp.CLIENT_MAIN_SET: self.set_main_settings,
            tcp.CLIENT_SYS_SET: self.set_system_settings,
            tcp.CLIENT_FTP: self.receive_ftp,
            tcp.CLIENT_LOAD_SYSTEM: self.refresh_system,
            tcp.CLIENT_RQST_TRACE: self.ray_trace_for_client,
            tcp.CLIENT_PARAMS: self.receive_parameters,
        }

    def got_connection(self):
        client_socket = tcp_base.TcpDataPacker(
            self.nextPendingConnection(),
            tcp_base.SERVER_ACK,
            tcp_base.CLIENT_ACK,
            verbose=True
        )
        if self.client_socket is None:
            self.client_socket = client_socket
            self.client_socket.data_ready.sig.connect(self.data_received)
            self.client_socket.errorOccurred.connect(self.socket_closed_remotely)
            self.client_socket.disconnected.connect(self.socket_closed_remotely)
        else:
            # An incoming connection was made, but we already have a connection
            client_socket.write(tcp_base.SERVER_FULL)
            client_socket.write(tcp_base.SERVER_FULL)
            client_socket.close()

    def data_received(self, header, data):
        try:
            self.message_LUT[header](data)
        except Exception:
            self.send_nonfatal_error("receiving data from client.")

    def socket_closed_remotely(self, _):
        self.client_socket = None
        self.cached_component_settings = None
        self.optical_system = None
        print("Connection closed from client side.")
        self.clean_temp()

    def send_nonfatal_error(self, context):
        ex = str(traceback.format_exc())
        self.client_socket.write(tcp.SERVER_ERROR, pickle.dumps((context, ex)))
        print("Nonfatal exception raised: " + context)
        print(ex)
        print("--End traceback--")

    def set_main_settings(self, data):
        try:
            self.settings.update(pickle.loads(data))
            self.client_socket.write(tcp.SERVER_M_SET_ACK, tcp_base.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_M_SET_ACK, tcp_base.FALSE)
            self.send_nonfatal_error("updating main settings")

    def set_system_settings(self, data):
        try:
            data = pickle.loads(data)
            # These settings may contain paths that need to be shifted to the correct location.
            # Client helpfully packages the string $PATH$ in the place where the server's temp folder path needs to go.
            for component_settings in data.values():
                for k, v in component_settings.items():
                    if type(v) is str and "$PATH$" in v:
                        local_path = v.replace("$PATH$", "")
                        component_settings[k] = str(self.temp_path / local_path)

            # Save the settings into a file.  They will be applied once refresh_system is called
            settings.Settings(**data).save(str(self.temp_path / "settings.data"))

            self.client_socket.write(tcp.SERVER_S_SET_ACK, tcp_base.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_S_SET_ACK, tcp_base.FALSE)
            self.send_nonfatal_error("updating system settings")

    def receive_ftp(self, data):
        local_filename = "FAILED_TO_UNPICKLE_DATA"
        try:
            local_filename, file_stream = pickle.loads(data)
            full_path = self.temp_path / local_filename
            self._write_file(full_path, file_stream)
            # note that this file has been freshly received
            self.received_files[local_filename] = True
            self.client_socket.write(tcp.SERVER_FTP_ACK, pickle.dumps((True, local_filename)))
        except Exception:
            self.client_socket.write(tcp.SERVER_FTP_ACK, pickle.dumps((False, local_filename)))
            self.send_nonfatal_error(f"receiving file {local_filename}")

    @staticmethod
    def _write_file(full_path, file_stream):
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(full_path), "wb") as outFile:
            outFile.write(file_stream)

    def refresh_system(self, _):
        try:
            # Check whether the optical_script is freshly received
            if self.received_files["optical_script.py"]:
                # the optical script is stale, so it needs to be loaded
                self.received_files["optical_script.py"] = False

                # Load the script as a module
                spec = importlib.util.spec_from_file_location(
                    name="system_module",
                    location=Path(self.settings.system_path) / "optical_script.py"
                )
                system_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(system_module)

                # Call the module to build the optical system.
                self.optical_system = system_module.get_system(self)

                # If we have received cached parameters, apply them to the system.
                if self.cached_parameters is not None:
                    for key, value in self.cached_parameters.items():
                        self.optical_system.parts[key].param_assign(value)
            else:
                # the optical script is fresh, so just update the settings and reload stale supporting files
                # If we already have an optical system, then we can update its settings directly
                # Confusing bug... when this line was originally settings.load, the settings would not change,
                # because the connection between the parts' settings and the system settings would be broken,
                # because settings.load creates whole new dictionaries.  Need to keep the dictionaries and
                # update them instead.
                self.optical_system.settings.reload(str(self.temp_path / "settings.data"))
                for k, v in self.received_files.items():
                    if v:
                        k = Path(k)
                        component_name = k.parent
                        file_type = k.stem
                        full_path = self.temp_path / k
                        self.reload_file(file_type, component_name, full_path)

                # So... sources cache certain properties (rotation) outside of settings, meaning that simply updating
                # the settings does not work.  So I need to call a new function on each component to let it know to
                # refresh itself from settings.
                for component in self.optical_system.parts.values():
                    try:
                        component.refresh_from_settings()
                    except AttributeError:
                        pass


            # In both branches above, the supported files will get loaded, either explicitly in the else clause
            # or while the system is loaded in the if clause.  So always need to mark every received file as
            # fresh here
            for k in self.received_files.keys():
                self.received_files[k] = False

            self.client_socket.write(tcp.SERVER_SYS_L_ACK, tcp_base.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_SYS_L_ACK, tcp_base.FALSE)
            self.send_nonfatal_error("refreshing system")

    def reload_file(self, file_type, component_name, full_path):
        file_type, component_name = str(file_type), str(component_name)
        try:
            component = self.optical_system.parts[component_name]
            if file_type == "spectrum":
                component.make_wavelengths.load_spectrum()
            elif file_type == "mesh_input":
                component.load(full_path)
            else:
                raise ValueError(f"Tracing_TCP_Server.reload_file: unknown file_type {file_type} was specified.")
        except Exception:
            self.send_nonfatal_error("reloading file")

    def clean_temp(self):
        for f in self.temp_path.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except PermissionError:
                    pass
            else:
                for g in f.iterdir():
                    try:
                        g.unlink()
                    except PermissionError:
                        pass
                f.rmdir()
        self.received_files.clear()

    def ray_trace_for_client(self, _):
        try:
            self.optical_system.update()
            self.optical_system.ray_trace(self.settings.trace_depth)
            all_rays = self.optical_system.all_rays.numpy()
            self.client_socket.write(tcp.SERVER_TRACE_RSLT, pickle.dumps(all_rays))
        except Exception:
            self.client_socket.write(tcp.SERVER_TRACE_RSLT, pickle.dumps(np.zeros((0, 7), dtype=np.float64)))
            self.send_nonfatal_error("ray trace for client")

    def receive_parameters(self, data):
        try:
            if self.optical_system:
                for key, value in pickle.loads(data).items():
                    self.optical_system.parts[key].param_assign(value)
            else:
                self.cached_parameters = pickle.loads(data)
            self.client_socket.write(tcp.SERVER_PARAM_ACK, tcp_base.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_PARAM_ACK, tcp_base.FALSE)
            self.send_nonfatal_error("receiving parameters")


if __name__ == "__main__":
    # Global exception hook to pick up on exceptions thrown in worker threads.
    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook
    app = qtw.QApplication([])
    server = TraceServer(tcp.DEFAULT_PORT)
    app.aboutToQuit.connect(server.clean_temp)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec_()
