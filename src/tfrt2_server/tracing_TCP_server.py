# Disable redundant TF warnings
if __name__ == "__main__":
    import logging
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
from pathlib import Path
import pickle
import importlib.util
import traceback
import signal
import argparse

import numpy as np
import PyQt5.QtNetwork as qtn
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc

import tfrt2.tcp_base as tcp
import tfrt2.settings as settings
from tfrt2.performance_tracer import PerformanceTracer


class TraceServer(qtn.QTcpServer):
    driver_type = "server"
    temp_path = Path(__file__).parent.resolve() / "server_temp"

    def __init__(self, port, args):
        super().__init__()
        self.optical_system = None
        self.listen(port=port)
        self.newConnection.connect(self.got_connection)
        self.client_socket = None
        self.settings = settings.Settings(system_path=str(self.temp_path))
        self.cached_component_settings = None
        self.cached_parameters = None
        self.received_files = {}
        self.busy_signal = BusySignal()
        self.busy_signal.sig.connect(self.send_busy_signal)
        self._send_data = DataSignal()
        self._send_data.sig.connect(self._send_data_callback)

        self.clean_temp()

        self.message_LUT = {
            tcp.CLIENT_MAIN_SET: self.set_main_settings,
            tcp.CLIENT_SYS_SET: self.set_system_settings,
            tcp.CLIENT_FTP: self.receive_ftp,
            tcp.CLIENT_LOAD_SYSTEM: self.refresh_system,
            tcp.CLIENT_RQST_TRACE: self.ray_trace_for_client,
            tcp.CLIENT_PARAMS: self.receive_parameters,
            tcp.CLIENT_RQST_ILUM: self.measure_illuminance,
            tcp.CLIENT_RESET_SYS: self.reset_system,
            tcp.CLIENT_ABORT_PROCS: self.abort_processing
        }

        # Make a performance tracer, which will organize the process of doing various high-throughput computing
        # tasks on the optical system.
        self.engine = PerformanceTracer(self, None, args.device_count)

    def got_connection(self):
        client_socket = tcp.TcpDataPacker(
            self.nextPendingConnection(),
            tcp.SERVER_ACK,
            tcp.CLIENT_ACK,
            verbose=True
        )
        if self.client_socket is None:
            self.client_socket = client_socket
            self.client_socket.data_ready.sig.connect(self.data_received)
            self.client_socket.disconnected.connect(self.socket_closed_remotely)
            self.client_socket.connection_validated.sig.connect(self.got_validation)
        else:
            # An incoming connection was made, but we already have a connection
            client_socket.write(tcp.SERVER_FULL)
            client_socket.write(tcp.SERVER_FULL)
            client_socket.close()

    def got_validation(self):
        self.client_socket.write(tcp.SERVER_MESSAGE, pickle.dumps(self.engine.system_report()))
        self.client_socket.write(tcp.SERVER_READY)

    def data_received(self, header, data):
        try:
            self.message_LUT[header](data)
        except Exception:
            self.send_nonfatal_error("receiving data from client.")

    def socket_closed_remotely(self, _=None):
        self.client_socket = None
        self.cached_component_settings = None
        self.cached_parameters = None
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
        if self.engine.try_wait():
            self.client_socket.write(tcp.SERVER_M_SET_ACK, tcp.FALSE)
            self.send_nonfatal_error("updating main settings: server was busy")
            return

        try:
            self.settings.update(pickle.loads(data))
            self.client_socket.write(tcp.SERVER_M_SET_ACK, tcp.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_M_SET_ACK, tcp.FALSE)
            self.send_nonfatal_error("updating main settings")

    def set_system_settings(self, data):
        if self.engine.try_wait():
            self.client_socket.write(tcp.SERVER_S_SET_ACK, tcp.FALSE)
            self.send_nonfatal_error("updating system settings: server was busy")
            return

        try:
            data = pickle.loads(data)
            # These settings may contain paths that need to be shifted to the correct location.
            # Client helpfully packages the string $PATH$ in the place where the server's temp folder path needs to go.
            for component_settings in data.values():
                for k, v in component_settings.items():
                    if type(v) is str and "$PATH$" in v:
                        component_settings[k] = str(self.ftp_to_path(v.replace("$PATH$", "")))

            # Save the settings into a file.  They will be applied once refresh_system is called
            settings.Settings(**data).save(str(self.temp_path / "settings.data"))

            self.client_socket.write(tcp.SERVER_S_SET_ACK, tcp.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_S_SET_ACK, tcp.FALSE)
            self.send_nonfatal_error("updating system settings")

    def receive_ftp(self, data):
        local_filename = "FAILED_TO_UNPICKLE_DATA"
        if self.engine.try_wait():
            self.client_socket.write(tcp.SERVER_FTP_ACK, pickle.dumps((False, local_filename)))
            self.send_nonfatal_error(f"receiving file {local_filename}: server was busy.")
            return

        try:
            local_filename, file_stream = pickle.loads(data)
            try:
                full_path = self.ftp_to_path(local_filename)
            except ValueError:
                full_path = self.temp_path / local_filename
            self._write_file(full_path, file_stream)
            # note that this file has been freshly received
            self.received_files[local_filename] = True
            self.client_socket.write(tcp.SERVER_FTP_ACK, pickle.dumps((True, local_filename)))
        except Exception:
            self.client_socket.write(tcp.SERVER_FTP_ACK, pickle.dumps((False, local_filename)))
            self.send_nonfatal_error(f"receiving file {local_filename}")

    def ftp_to_path(self, local_filename, get_parts=False):
        component, file_name, suffix = str(local_filename).split('|')
        if get_parts:
            return file_name, component, (self.temp_path / component / file_name).with_suffix(suffix)
        else:
            return (self.temp_path / component / file_name).with_suffix(suffix)

    @staticmethod
    def _write_file(full_path, file_stream):
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(full_path), "wb") as outFile:
            outFile.write(file_stream)

    def refresh_system(self, _):
        if self.engine.try_wait():
            self.client_socket.write(tcp.SERVER_SYS_L_ACK, tcp.FALSE)
            self.send_nonfatal_error("refreshing system: server was busy")
            return

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
                self.optical_system.post_init()
                self.engine.optical_system = self.optical_system

                # If we have received cached parameters, apply them to the system.
                if self.cached_parameters is not None:
                    for key, value in self.cached_parameters.items():
                        self.optical_system.parts[key].param_assign(value)
                self.cached_parameters = None

            else:
                # the optical script is fresh, so just update the settings.
                # Confusing bug... when this line was originally settings.load, the settings would not change,
                # because the connection between the parts' settings and the system settings would be broken,
                # because settings.load creates whole new dictionaries.  Need to keep the dictionaries and
                # update them instead.
                self.optical_system.settings.reload(str(self.temp_path / "settings.data"))

            # My initial intuition was to only run this code in the else clause above, but it may need to be run always
            # because I am finding that the system is only ever loading the default spectrum, though it can load a
            # correct updated one after a secondary synchronization.
            for k, v in self.received_files.items():
                if v:
                    self.reload_file(k)

            # So... sources cache certain properties (rotation) outside of settings, meaning that simply updating
            # the settings does not work.  So I need to call a new function on each component to let it know to
            # refresh itself from settings.
            for component in self.optical_system.parts.values():
                try:
                    component.refresh_from_settings()
                except AttributeError:
                    pass

            self.client_socket.write(tcp.SERVER_SYS_L_ACK, tcp.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_SYS_L_ACK, tcp.FALSE)
            self.send_nonfatal_error("refreshing system")

    def reload_file(self, file_key):
        if self.engine.try_wait():
            self.send_nonfatal_error("reloading file, server was busy")
            return

        file_type, component_name, full_path = self.ftp_to_path(file_key, get_parts=True)
        try:
            component = self.optical_system.parts[component_name]
            if file_type == "spectrum":
                try:
                    component.make_wavelengths.load_spectrum()
                except AttributeError:
                    # It isn't a spectrum object, so don't need to do anything
                    pass
            elif file_type == "mesh_input":
                component.load(full_path)
            else:
                raise ValueError(f"Tracing_TCP_Server.reload_file: unknown file_type {file_type} was specified.")

            self.received_files[file_key] = False
        except Exception:
            self.send_nonfatal_error("reloading file")

    def clean_temp(self):
        # Make sure the temp folder exists
        if not self.temp_path.exists():
            self.temp_path.mkdir(exist_ok=True)

        # Empty it
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
        self.engine.queue_job("full_ray_trace")

    def receive_parameters(self, data):
        if self.engine.try_wait():
            self.client_socket.write(tcp.SERVER_PARAM_ACK, tcp.FALSE)
            self.send_nonfatal_error("receiving parameters, server was busy")
            return

        try:
            if self.optical_system:
                for key, value in pickle.loads(data).items():
                    self.optical_system.parts[key].param_assign(value)
            else:
                self.cached_parameters = pickle.loads(data)
            self.client_socket.write(tcp.SERVER_PARAM_ACK, tcp.TRUE)
        except Exception:
            self.client_socket.write(tcp.SERVER_PARAM_ACK, tcp.FALSE)
            self.send_nonfatal_error("receiving parameters")

    def measure_illuminance(self, data):
        settings = pickle.loads(data)
        try:
            ray_count, x_res, y_res, x_min, x_max, y_min, y_max = settings
            standalone_plot = False
        except TypeError:
            ray_count = self.optical_system.settings.goal.flattener_ray_requirement
            x_res = self.optical_system.settings.goal.f_c1_res
            y_res = self.optical_system.settings.goal.f_c2_res
            x_min = self.optical_system.settings.goal.c1_min
            x_max = self.optical_system.settings.goal.c1_max
            y_min = self.optical_system.settings.goal.c2_min
            y_max = self.optical_system.settings.goal.c2_max

        self.engine.queue_job(
            "illuminance_plot",
            (ray_count, True, x_res, y_res, x_min, x_max, y_min, y_max)
        )

    def reset_system(self, _):
        self.optical_system = None
        self.cached_component_settings = None
        self.cached_parameters = None
        self.clean_temp()
        self.engine.abort()
        self.client_socket.write(tcp.SERVER_SYS_RST_ACK)

    def send_busy_signal(self, state):
        if state:
            self.client_socket.write(tcp.SERVER_BUSY)
        else:
            self.client_socket.write(tcp.SERVER_READY)

    def abort_processing(self, _):
        self.engine.abort_jobs()

    def _send_data_callback(self, data):
        header, data = data
        self.client_socket.write(header, data)

    def send_data(self, header, data=None):
        self._send_data.sig.emit((header, data))


class BusySignal(qtc.QObject):
    sig = qtc.pyqtSignal(bool)


class DataSignal(qtc.QObject):
    sig = qtc.pyqtSignal(tuple)


if __name__ == "__main__":
    # process command line arguments
    parser = argparse.ArgumentParser(description="Run tfrt2's tracing server")
    parser.add_argument(
        "--devices", default=-1, type=int, dest="device_count",
        help="How many processes to use for tracing.  Defaults to -1, in which case the engine decides for itself."
    )

    # Global exception hook to pick up on exceptions thrown in worker threads.
    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook
    app = qtw.QApplication([])
    server = TraceServer(tcp.DEFAULT_PORT, parser.parse_args())
    app.aboutToQuit.connect(server.clean_temp)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec_()
