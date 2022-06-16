import sys
import importlib

import PyQt5.QtWidgets as qtw
import pyvistaqt as pvqt


class OpticClientWindow(qtw.QWidget):
    def __init__(self):
        super().__init__(windowTitle="Linear Mark 5 Dev Client")

        # Load the system, by looking at the command line arguments
        if len(sys.argv) >= 2:
            try:
                self.system_module = importlib.import_module(sys.argv[1])
                self.system = self.system_module.get_system()
            except ImportError as e:
                raise ImportError("OpticClient: Could not import the system module {sys.argv[1]}") from e
        else:
            raise RuntimeError("OpticClient: Must specify path to the system defining module in command line")

        self.plot = None
        self.control_panes = [
            ("Display Settings", DisplayControls()),
            ("Parameters", qtw.QWidget()),
            ("Optimization", qtw.QWidget()),
        ]
        self.pane_stack = None
        self.build_ui()

    def build_ui(self):
        main_layout = qtw.QGridLayout()
        self.setLayout(main_layout)

        # Setup the control layout
        control_layout = qtw.QVBoxLayout()
        main_layout.addLayout(control_layout, 0, 0)

        tab_selector = qtw.QComboBox()
        tab_selector.insertItems(0, [pane[0] for pane in self.control_panes])
        tab_selector.currentIndexChanged.connect(self.change_active_pane)
        control_layout.addWidget(tab_selector)

        self.pane_stack = qtw.QStackedWidget()
        for pane in self.control_panes:
            self.pane_stack.addWidget(pane[1])
        self.pane_stack.setMaximumWidth(350)
        control_layout.addWidget(self.pane_stack)

        # Setup the main 3D plot
        self.plot = pvqt.QtInteractor()
        self.plot.add_axes()
        main_layout.addWidget(self.plot, 0, 1)

    def change_active_pane(self, i):
        print(f"selected item {i}")
        self.pane_stack.setCurrentIndex(i)

    def quit(self):
        pass


class DisplayControls(qtw.QWidget):
    pass


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
