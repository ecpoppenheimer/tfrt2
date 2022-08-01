import pickle

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

DEFAULT_PORT = 11170  # The default port to use for TCP connections

HEADER_SIZE = 16  # The size in bytes of an identifier that identifies the kind of message being sent
INT_SIZE = 4  # The number of bytes to use to store an integer, which is used to store the size of each data chunk.

# Some prepacked bytes objects that will be used frequently enough to deserve to be precompiled
TRUE = pickle.dumps(True)
FALSE = pickle.dumps(False)
DATA_SIZE_ZERO = int(0).to_bytes(INT_SIZE, byteorder='little', signed=False)

#                    b"1234123412341234"
CLIENT_ACK =         b"tfrt2 client con"
SERVER_ACK =         b"tfrt2 server con"
SERVER_FULL =        b"connect already "
SERVER_ERROR =       b"server error    "
CLIENT_MAIN_SET =    b"client main set "
SERVER_M_SET_ACK =   b"server m set ack"
CLIENT_SYS_SET =     b"client sys set  "
SERVER_S_SET_ACK =   b"server s set ack"
CLIENT_FTP =         b"client ftp      "
SERVER_FTP_ACK =     b"server ftp ack  "
CLIENT_LOAD_SYSTEM = b"client load syst"
SERVER_SYS_L_ACK   = b"server sys l ack"
CLIENT_RQST_TRACE  = b"client rqs trace"
SERVER_TRACE_RSLT  = b"server trace rst"
CLIENT_PARAMS      = b"client parameter"
SERVER_PARAMS      = b"server parameter"
SERVER_PARAM_ACK   = b"server param ack"


class InvalidHeaderError(ValueError):
    def __init__(self, header):
        super().__init__()
        self._h = str(header)

    def __str__(self):
        return self._h


class DataReady(qtc.QObject):
    sig = qtc.pyqtSignal(bytes, bytes)


class ConnectionValidated(qtc.QObject):
    sig = qtc.pyqtSignal(bool)


class IP4Validator(qtg.QValidator):
    """
    Validator to check the validity of the IP address.  This code was copied with minor modifications from
    https://stackoverflow.com/questions/53873737/pyqt5-qline-setinputmask-setvalidator-ip-address
    """
    def __init__(self, parent=None):
        super(IP4Validator, self).__init__(parent)

    def validate(self, address, pos):
        if not address:
            return qtg.QValidator.Acceptable, address, pos

        # check to permit 'localhost'
        if address == "localhost":
            return qtg.QValidator.Acceptable, address, pos
        elif address in "localhost":
            return qtg.QValidator.Intermediate, address, pos

        octets = address.split(".")
        size = len(octets)
        if size > 4:
            return qtg.QValidator.Invalid, address, pos
        empty_octet = False
        for octet in octets:
            if not octet or octet == "___" or octet == "   ":  # check for mask symbols
                empty_octet = True
                continue
            try:
                value = int(str(octet).strip(' _'))  # strip mask symbols
            except Exception:
                return qtg.QValidator.Intermediate, address, pos
            if value < 0 or value > 255:
                return qtg.QValidator.Invalid, address, pos
        if size < 4 or empty_octet:
            return qtg.QValidator.Intermediate, address, pos
        return qtg.QValidator.Acceptable, address, pos


class TcpDataPacker:
    """
    A simple wrapper for a PyQt QTcpSocket that manages splitting and merging individual messages into complete chunks.

    QTcpSocket manages a stream, and apparently handles arranging and error checking the data it receives, but will
    fire it's readReady signal even if a whole message has yet to be transmitted; this class will take care of this
    and so will emit full, completed chunks rather than segments of the stream.
    """
    def __init__(self, socket, send_validator, receive_validator, verbose=False):
        super().__init__()
        self._socket = socket
        self._receive_validator = receive_validator
        self._required_bytes = 0
        self._chunk_in_progress = False
        self._socket.readyRead.connect(self._read_some)
        self.validated = False
        self.verbose = verbose

        # Public PyQt signals
        self.data_ready = DataReady()
        self.connection_validated = ConnectionValidated()

        # Send the validation message
        self._socket.write(send_validator)

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def _read_some(self):
        print(f"read some called")
        while self._socket.bytesAvailable() >= INT_SIZE:
            if self.validated:
                print(f"performing a main read with {self._socket.bytesAvailable()} available bytes")
                if not self._chunk_in_progress:
                    self._chunk_in_progress = True
                    self._required_bytes = int.from_bytes(self._socket.read(INT_SIZE), byteorder='little', signed=False)

                # required_bytes is the size of the data chunk, but we also have to be able to read off the header.
                # If we don't have enough data to do both, break the loop.
                if self._socket.bytesAvailable() < self._required_bytes + HEADER_SIZE:
                    print("chunk incomplete")
                    break
                else:
                    # Have enough data to read off a header and data chunk.
                    print("chunk obtained...")
                    self._chunk_in_progress = False
                    header = self._socket.read(HEADER_SIZE)
                    data = self._socket.read(self._required_bytes)
                    self.data_ready.sig.emit(header, data)
            else:
                # The very first message must be at least size tcp.HEADER_SIZE and must start with the
                # receive_validator, or the protocol will terminate the connection
                if not self._check_validation():
                    # If validation did not pass, break the loop.
                    break

    def _check_validation(self):
        """
        Check that the validation message was received.  Will return True if validated, and will return False
        if the buffer does not have enough data to validate.  If the validation fails, will raise a ValidationError.
        """
        if self._socket.bytesAvailable() >= HEADER_SIZE:
            header = self._socket.read(HEADER_SIZE)
            if header == self._receive_validator:
                self.validated = True
                self.connection_validated.sig.emit(True)
                if self.verbose:
                    print("Established connection!")
                return True
            else:
                self.connection_validated.sig.emit(False)
                self._socket.abort()
                return False
        else:
            # Not a large enough amount of data was received to do the validation
            return False

    def write(self, header, data=None):
        if data is not None:
            data_size = len(data)
            self._socket.write(data_size.to_bytes(INT_SIZE, byteorder='little', signed=False) + header + data)
        else:
            self._socket.write(DATA_SIZE_ZERO + header)
