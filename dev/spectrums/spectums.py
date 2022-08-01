from pathlib import Path

import numpy as np

import tfrt2.wavelength as wv

path_root = Path(__file__).parent

limits = (wv.VISIBLE_MIN, wv.VISIBLE_MAX)
x = np.linspace(limits[0], limits[1], 7)
y = [0, 1, 3, 2, 4, 1, 0]

spectrum = wv.Spectrum((x, y), limits)
spectrum.save(path_root / "test_spikey.dat")

# ----------------------------------------------------------------------------------------------------------------------

limits = (wv.VISIBLE_MIN, wv.VISIBLE_MAX)
x = limits
y = (1, 1)

spectrum = wv.Spectrum((x, y), limits)
spectrum.save(path_root / "uniform.dat")

# ----------------------------------------------------------------------------------------------------------------------

limits = (500, 520)
x = np.linspace(limits[0], limits[1], 7)
y = [1, 4, 12, 15, 12, 4, 1]

spectrum = wv.Spectrum((x, y), limits)
spectrum.save(path_root / "test_yellow.dat")
#spectrum.standalone_plot()
