import pathlib

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import numpy as np

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class SettingsEntryBox(qtw.QWidget):
    def __init__(
            self, settings, key, value_type, validator=None, callback=None, label=None
    ):
        super().__init__()
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        if label is None:
            self.label = qtw.QLabel(str(key).replace("_", " "))
        else:
            self.label = qtw.QLabel(str(label))
        self.label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)
        layout.addWidget(self.label)

        self.edit_box = qtw.QLineEdit()
        layout.addWidget(self.edit_box)
        self.edit_box.setText(str(settings.dict[key]))
        if validator:
            self.edit_box.setValidator(validator)

        def edit_callback():
            value = value_type(self.edit_box.text())
            settings.dict[key] = value

        self.edit_box.editingFinished.connect(edit_callback)
        if callback is not None:
            try:
                for each in callback:
                    self.edit_box.editingFinished.connect(each)
            except TypeError:
                self.edit_box.editingFinished.connect(callback)


class SettingsRangeBox(qtw.QWidget):
    def __init__(self, settings, label, low_key, high_key, value_type, validator=None, callback=None):
        super().__init__()
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.settings = settings
        self.value_type = value_type
        self.low_key = low_key
        self.high_key = high_key
        self.callback = callback

        self.label = qtw.QLabel(label)
        self.label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)
        layout.addWidget(self.label)

        self.low_entry = qtw.QLineEdit()
        layout.addWidget(self.low_entry)
        self.low_entry.setText(str(settings.dict[low_key]))
        if validator:
            self.low_entry.setValidator(validator)

        self.high_entry = qtw.QLineEdit()
        layout.addWidget(self.high_entry)
        self.high_entry.setText(str(settings.dict[high_key]))
        if validator:
            self.high_entry.setValidator(validator)

        self.low_entry.editingFinished.connect(self.low_callback)
        self.high_entry.editingFinished.connect(self.high_callback)

    def low_callback(self):
        low_value = self.value_type(self.low_entry.text())
        high_value = self.value_type(self.high_entry.text())

        if low_value >= high_value:
            self.low_entry.setStyleSheet("QLineEdit { background-color: pink}")
        else:
            self.common_callback(low_value, high_value)

    def high_callback(self):
        low_value = self.value_type(self.low_entry.text())
        high_value = self.value_type(self.high_entry.text())

        if high_value <= low_value:
            self.high_entry.setStyleSheet("QLineEdit { background-color: pink}")
        else:
            self.common_callback(low_value, high_value)

    def common_callback(self, low_value, high_value):
        self.high_entry.setStyleSheet("QLineEdit { background-color: white}")
        self.low_entry.setStyleSheet("QLineEdit { background-color: white}")
        self.settings.dict[self.low_key] = low_value
        self.settings.dict[self.high_key] = high_value
        if self.callback is not None:
            try:
                for each in self.callback:
                    each()
            except TypeError:
                self.callback()

    def set_range(self, low, high):
        self.low_entry.setText(str(low))
        self.low_callback()
        self.high_entry.setText(str(high))
        self.high_callback()


class SettingsFileBox(qtw.QWidget):
    def __init__(
            self, settings, key, system_path, filter="*", mode=None, callback=None, callback_2=None
    ):
        super().__init__()
        self.filter = filter
        self.system_path = system_path
        if mode == "save":
            self.do_save = True
            self.do_load = False
            self.do_save_dialog_type = True
            self.save_callback = callback
            self.load_callback = None
        elif mode == "load":
            self.do_save = False
            self.do_load = True
            self.do_save_dialog_type = False
            self.save_callback = None
            self.load_callback = callback
        elif mode == "both":
            self.do_save = True
            self.do_load = True
            self.do_save_dialog_type = True
            self.save_callback = callback
            self.load_callback = callback_2
        elif mode == "none" or mode is None:
            self.do_save = False
            self.do_load = False
            self.do_save_dialog_type = True
            self.save_callback = None
            self.load_callback = None
        else:
            raise ValueError("SettingsFileBox: mode must be specified, and one of {save, load, both, none}")
        self.settings = settings
        self.key = key

        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        if self.do_save:
            self.save_button = qtw.QPushButton("Save")
            self.save_button.setMaximumWidth(32)
            self.save_button.clicked.connect(self.save)
            layout.addWidget(self.save_button)
        if self.do_load:
            self.load_button = qtw.QPushButton("Load")
            self.load_button.setMaximumWidth(32)
            self.load_button.clicked.connect(self.load)
            layout.addWidget(self.load_button)

        self.select_button = qtw.QPushButton("Select")
        self.select_button.setMaximumWidth(37)
        self.select_button.clicked.connect(self.select)
        layout.addWidget(self.select_button)

        self.label = qtw.QLineEdit()
        self.label.setText(str(self.settings.dict[key]))
        self.label.setReadOnly(True)
        layout.addWidget(self.label)

    def save(self):
        if self.save_callback is not None:
            try:
                self.save_callback()
            except TypeError:
                for each in self.save_callback:
                    each()

    def load(self):
        if self.load_callback is not None:
            try:
                self.load_callback()
            except TypeError:
                for each in self.load_callback:
                    each()

    def select(self):
        if self.do_save:
            selected_file, _ = qtw.QFileDialog.getSaveFileName(
                directory=str(self.system_path), filter=self.filter
            )
        else:
            selected_file, _ = qtw.QFileDialog.getOpenFileName(
                directory=str(self.system_path), filter=self.filter
            )
        if selected_file:
            self.settings.dict[self.key] = str(pathlib.Path(selected_file))
            self.label.setText(selected_file)
        self.label.setStyleSheet("QLineEdit { background-color: white}")

    def notify_bad_selection(self):
        self.label.setStyleSheet("QLineEdit { background-color: pink}")


class SettingsComboBox(qtw.QWidget):
    def __init__(self, settings, label, settings_key, settings_options, callback=None):
        super().__init__()
        self.settings = settings
        self.settings_key = settings_key
        self.settings_options = settings_options

        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel(label))

        selector = qtw.QComboBox()
        layout.addWidget(selector)
        selector.addItems(settings_options)
        selector.setCurrentIndex(settings_options.index(self.settings.dict[settings_key]))
        selector.currentIndexChanged.connect(self.set_setting)

        if callback is not None:
            try:
                for each in callback:
                    selector.currentIndexChanged.connect(each)
            except TypeError:
                selector.currentIndexChanged.connect(callback)

    def set_setting(self, index):
        self.settings.dict[self.settings_key] = self.settings_options[index]


class SettingsVectorBox(qtw.QWidget):
    def __init__(self, settings, label, settings_key, callback=None):
        super().__init__()
        self.settings = settings
        self.settings_key = settings_key

        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(qtw.QLabel(label))
        self.entries = []
        for i in range(3):
            initial = self.settings.dict[settings_key][i]
            entry = qtw.QLineEdit()
            self.entries.append(entry)
            entry.setText(str(initial))
            entry.setValidator(qtg.QDoubleValidator(-1e6, 1e6, 7))
            layout.addWidget(entry)

        self.entries[0].editingFinished.connect(self.callback_x)
        self.entries[1].editingFinished.connect(self.callback_y)
        self.entries[2].editingFinished.connect(self.callback_z)

        if callback is not None:
            try:
                for each in callback:
                    self.entries[0].editingFinished.connect(each)
                    self.entries[1].editingFinished.connect(each)
                    self.entries[2].editingFinished.connect(each)
            except TypeError:
                self.entries[0].editingFinished.connect(callback)
                self.entries[1].editingFinished.connect(callback)
                self.entries[2].editingFinished.connect(callback)

    def callback_x(self):
        value = float(self.entries[0].text())
        self.settings.dict[self.settings_key][0] = value

    def callback_y(self):
        value = float(self.entries[1].text())
        self.settings.dict[self.settings_key][1] = value

    def callback_z(self):
        value = float(self.entries[2].text())
        self.settings.dict[self.settings_key][2] = value


class SettingsCheckBox(qtw.QWidget):
    def __init__(self, settings, key, label, callback=None):
        super().__init__()

        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = qtw.QLabel(label)
        layout.addWidget(self.label)

        check_box = qtw.QCheckBox()
        layout.addWidget(check_box)
        check_box.setCheckState(settings.dict[key])
        check_box.setTristate(False)

        def set_setting(new_state):
            settings.dict[key] = bool(new_state)

        check_box.stateChanged.connect(set_setting)

        if callback is not None:
            try:
                for each in callback:
                    check_box.stateChanged.connect(each)
            except TypeError:
                check_box.stateChanged.connect(callback)


class ColorEntryButton(qtw.QPushButton):
    def __init__(self, settings, key, callback=None):
        super().__init__()
        self.setText("Color")
        self.clicked.connect(self.click)
        self.callback = callback
        self.settings = settings
        self.key = key

    def click(self):
        color = qtw.QColorDialog.getColor().name()
        self.settings.dict[self.key] = color
        self.setStyleSheet(f"QPushButton {{ background-color: {color}}}")
        self.callback()


class DelayedSlider(qtw.QSlider):
    def __init__(self, orientation=qtc.Qt.Orientation.Horizontal, time_delay=.05):
        super().__init__(orientation)

        self._time_delay_ms = int(time_delay * 1000)
        self._has_time_deferred_call = False
        self._timer = qtc.QTimer()

        self.valueChanged.connect(self._emit_value_changed)
        self.valueChanged = DelayedSliderValueChanged()

    def _emit_value_changed(self):
        if not self._has_time_deferred_call:
            self._has_time_deferred_call = True
            self._timer.singleShot(self._time_delay_ms, self._emit_timed_value_changed)

    def _emit_timed_value_changed(self):
        self._has_time_deferred_call = False
        self.valueChanged.sig.emit(self.value())


class DelayedSliderValueChanged(qtc.QObject):
    sig = qtc.pyqtSignal(int)

    def connect(self, *args):
        self.sig.connect(*args)


# ======================================================================================================================


class MPLWidget(qtw.QWidget):
    """
    A pyqt widget that displays a mpl plot, with a built in toolbar.

    Parameters
    ----------
    alignment : 4-tuple of floats, optional
        The area in figure coordinates where the axes will sit.  Defaults to None, in which
        case the value used is chosen based on the value of parameter blank.  If blank is
        True, uses the value (0.0, 0.0, 1.0, 1.0) in which case the axes cover exactly the
        whole drawing area, which is great if you are displaying an image.  If blank is False,
        uses (.1, .1, .898, .898) which is a good starter option for plots with displayed
        labels, but you may want to tweak manually.  .9 has a tendency to cut off the line on
        the left size of the plot.
        The first two numbers anchor the lower left corner, the second two
        are a span.  The values are all relative to the size of the canvas, so between 0 and
        1.
    fig_size : 2-tuple of floats, optional
        The default size of the figure.  But since pyqt messes with widget size a lot this
        is more of a rough starting guess than an actual set value..
    blank : bool, optional
        If False, the default, draws the plot as normal.
        If True, the canvas will be a blank white square with no axes or anything, ideal
        for drawing.
    args and kwargs passed to qtw.QWidget constructor

    Public Properties
    -----------------
    fig : mpl figure
        A handle to the generated mpl figure.
    ax : mpl axes
        A handle to the generated axes, where you can draw/plot stuff.
    fig_canvas : mpl backend object
        The canvas object that plugs into pyqt.

    Public Methods
    --------------
    draw() :
        Draw or redraw the figure.
    """

    def __init__(
            self,
            name=None,
            alignment=None,
            fig_size=(2.5, 2.5),
            blank=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.fig = Figure(figsize=fig_size)
        self.fig_canvas = FigureCanvas(self.fig)
        if alignment is None:
            if blank:
                alignment = (0.0, 0.0, 1.0, 1.0)
            else:
                alignment = (.1, .1, .898, .898)
        self.ax = self.fig.add_axes(alignment)
        if blank:
            self.ax.set_frame_on(False)
            self.ax.axes.get_xaxis().set_visible(False)
            self.ax.axes.get_yaxis().set_visible(False)
            self.ax.axes.get_xaxis().set_major_locator(plt.NullLocator())
            self.ax.axes.get_yaxis().set_major_locator(plt.NullLocator())

        layout = qtw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if name:
            label = qtw.QLabel(name)
            label.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Fixed)
            layout.addWidget(label)
        layout.addWidget(self.fig_canvas)
        layout.addWidget(NavigationToolbar(self.fig_canvas, self))
        self.setLayout(layout)

    def draw(self):
        self.fig_canvas.draw()


class MplImshowWidget(MPLWidget):
    def __init__(
        self,
        initial_data=None,
        blank=False,
        *args,
        **kwargs
    ):
        super().__init__(blank=blank, *args, **kwargs)
        if initial_data is None:
            initial_data = np.zeros((1, 1))
        self.plot = self.ax.imshow(initial_data, origin="lower", cmap="gray")

    def set_data(self, data, extent):
        self.plot.set_data(data)
        self.plot.set_extent(extent)
        self.plot.set_clim(np.min(data), np.max(data))
        self.draw()
