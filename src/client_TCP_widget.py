import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg

import tfrt2.src.component_widgets as cw

DEFAULT_PORT = 11170


class Client_TCP_Widget(qtw.QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client

        # Establish settings defaults
        self.client.settings.establish_defaults(server_port=DEFAULT_PORT, server_ip="localhost")

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
        self.ip_widget = cw.SettingsEntryBox(
            self.client.settings, "server_ip", str, validator=IP4Validator(), callback=self.set_ip
        )
        layout.addWidget(self.ip_widget, 2, 1)

        # Status indicator  layout.addWidget(_, 2, 0)

    def click_connection_button(self):
        print("trying to connect...")

    def set_ip(self):
        print(f"accepted IP address {self.client.settings.server_ip}")


class IP4Validator(qtg.QValidator):
    """
    Validator to check the validity of the IP address.  This code was copied with minor modifications from
    https://stackoverflow.com/questions/53873737/pyqt5-qline-setinputmask-setvalidator-ip-address
    """
    def __init__(self, parent=None):
        super(IP4Validator, self).__init__(parent)

    def validate(self, address, pos):
        if not address:
            return qtg.QValidator.Acceptable, pos

        # check to permit 'localhost'
        if address == "localhost":
            print("BOO!")2534
            return qtg.QValidator.Acceptable, pos
        elif address in "localhost":
            return qtg.QValidator.Intermediate, pos

        octets = address.split(".")
        size = len(octets)
        if size > 4:
            return qtg.QValidator.Invalid, pos
        empty_octet = False
        for octet in octets:
            if not octet or octet == "___" or octet == "   ":  # check for mask symbols
                empty_octet = True
                continue
            try:
                value = int(str(octet).strip(' _'))  # strip mask symbols
            except Exception:
                return qtg.QValidator.Intermediate, pos
            if value < 0 or value > 255:
                return qtg.QValidator.Invalid, pos
        if size < 4 or empty_octet:
            return qtg.QValidator.Intermediate, pos
        return qtg.QValidator.Acceptable, pos
