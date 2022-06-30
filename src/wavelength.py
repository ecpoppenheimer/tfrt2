"""
Utilities associated with wavelength and color.

rgb() implementation Based on
  http://www.physics.sfasu.edu/astro/color/spectra.html
  RGB VALUES FOR VISIBLE WAVELENGTHS   by Dan Bruton (astro@tamu.edu)

The Public API of this module exposes one function, rgb() which coverts a wavelength in nanometers to an RGB color
value that drawers can interpret.

It also exposes some useful constants, like the wavelength limits and the wavelengths associated with various colors.
"""

import numpy as np


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
