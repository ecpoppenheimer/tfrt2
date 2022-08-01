import sys

import PyQt5.QtNetwork as qtn
import PyQt5.QtWidgets as qtw

from tfrt2.src.client_TCP_widget import DEFAULT_PORT


class EchoServer(qtn.QTcpServer):
    def __init__(self, port):
        super().__init__()
        self.listen(port=port)
        self.newConnection.connect(self.got_connection)

        self.client_socket = None

    def got_connection(self):
        if self.client_socket is None:
            self.client_socket = self.nextPendingConnection()
            self.client_socket.write(b"You just connected to an echo server!")
            self.client_socket.readyRead.connect(self.read_data)
            self.client_socket.errorOccurred.connect(self.socket_closed_remotely)
        else:
            print("something unexpected happened, another connection attempt was made.")

    def read_data(self):
        data = self.client_socket.read(self.client_socket.bytesAvailable())
        print(f"got some data {data}")
        self.client_socket.write(bytes(f"echo server is returning your data {data}", "utf-8"))

    def socket_closed_remotely(self, _):
        print("Client close the connection.")
        self.client_socket = None

if __name__ == "__main__":
    # Global exception hook to pick up on exceptions thrown in worker threads.
    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook
    app = qtw.QApplication([])
    server = EchoServer(DEFAULT_PORT)
    exit(app.exec_())
