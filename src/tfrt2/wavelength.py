"""
Utilities associated with wavelength and color.

rgb() implementation Based on
  http://www.physics.sfasu.edu/astro/color/spectra.html
  RGB VALUES FOR VISIBLE WAVELENGTHS   by Dan Bruton (astro@tamu.edu)

The Public API of this module exposes one function, rgb() which coverts a wavelength in nanometers to an RGB color
value that drawers can interpret.

It also exposes some useful constants, like the wavelength limits and the wavelengths associated with various colors.
"""
import pickle
import pathlib

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

import tfrt2.component_widgets as cw
import cumdistf.src.cumdistf.cdf as cdf


def _factor(wl):
    return np.select(
        [wl > 700.0, wl < 420.0, True],
        [
            0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0),
            0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0),
            1.0,
        ],
    )


def _raw_r(wl):
    return np.select(
        [wl >= 580.0, wl >= 510.0, wl >= 440.0, wl >= 380.0, True],
        [1.0, (wl - 510.0) / (580.0 - 510.0), 0.0, (wl - 440.0) / (380.0 - 440.0), 0.0],
    )


def _raw_g(wl):
    return np.select(
        [wl >= 645.0, wl >= 580.0, wl >= 490.0, wl >= 440.0, True],
        [0.0, (wl - 645.0) / (580.0 - 645.0), 1.0, (wl - 440.0) / (490.0 - 440.0), 0.0],
    )


def _raw_b(wl):
    return np.select(
        [wl >= 510.0, wl >= 490.0, wl >= 380.0, True],
        [0.0, (wl - 510.0) / (490.0 - 510.0), 1.0, 0.0],
    )


_gamma = 0.80
_ww = np.arange(380.0, 781.0)


def _correct_r(wl):
    return np.power(_factor(wl) * _raw_r(wl), _gamma)


def _correct_g(wl):
    return np.power(_factor(wl) * _raw_g(wl), _gamma)


def _correct_b(wl):
    return np.power(_factor(wl) * _raw_b(wl), _gamma)


def rgb():
    """
    To turn to a mpl colormap, use mpl.colors.ListedColormap(rgb())
    """
    return np.transpose([_correct_r(_ww), _correct_g(_ww), _correct_b(_ww)])


# Various useful wavelength related constants
VISIBLE_MIN = 380
VISIBLE_MAX = 780

RED = 680
ORANGE = 620
YELLOW = 575
GREEN = 510
BLUE = 450
PURPLE = 400

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]


class Spectrum(cdf.CumulativeDistributionFunction1D):
    def __init__(self, data, limits=(VISIBLE_MIN, VISIBLE_MAX), interp_res=100):
        """
        A class to sample wavelengths out of a spectrum.  Wavelengths are in nm, by default.

        Parameters
        ----------
        data : str or 2-tuple of 1D arrays.
            The data used to generate the spectrum.  If this is a str, it is interpreted as a filename that can be
            unpickled into a 2-tuple of 1D arrays.
        limits : 2-tuple of floats, optional.
            The wavelength limits.
        interp_res : int, optional.
            The resolution to use when interpolating the data.
        """
        self.x_data, self.y_data = None, None
        self.interp_res = interp_res
        self.limits = limits
        if type(data) in {str or pathlib.Path}:
            self.load(data)
        else:
            self.x_data, self.y_data = data
        super().__init__(limits)
        self.compute()

    def compute(self):
        interp = interp1d(self.x_data, self.y_data)
        super().compute(
            density=interp(np.linspace(self.x_min, self.x_max, self.interp_res)),
            direction="forward"
        )

    @staticmethod
    def load(filename):
        with open(filename, "rb") as inFile:
            x_data, y_data, limits, interp_res = pickle.load(inFile)
        return Spectrum((x_data, y_data), limits=limits, interp_res=interp_res)

    def save(self, filename):
        with open(filename, "wb") as outFile:
            pickle.dump((self.x_data, self.y_data, self.limits, self.interp_res), outFile, pickle.HIGHEST_PROTOCOL)

    def __call__(self, rnd):
        return super().__call__(rnd)

    def standalone_plot(self, y_res=100):
        fig, axis = plt.subplots()
        self.plot_to_axis(axis, y_res)
        plt.show()

    def plot_to_axis(self, ax, y_res):
        x = np.linspace(self.x_min, self.x_max, self._density.shape[0])
        y_max = np.amax(self._density)
        y = np.linspace(0, y_max, y_res)
        x_im, y = np.meshgrid(x, y)

        rtn = ax.imshow(
            x_im,
            cmap=mpl.colors.ListedColormap(rgb()),
            origin="lower",
            extent=(self.x_min, self.x_max, 0, y_max),
            clim=(VISIBLE_MIN, VISIBLE_MAX)
        )
        ax.set_ylim(0, y_max)
        ax.set_aspect("auto")
        ax.fill_between(x, self._density, y_max, color="w")

        return rtn

    def plot_to_widget(self, y_res=100, fig_size=(2.5, 2.5)):
        widget = cw.MPLWidget(fig_size=fig_size)
        rtn = self.plot_to_axis(widget.ax, y_res)
        widget.draw()

        return widget, rtn
