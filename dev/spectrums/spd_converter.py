import pathlib

import numpy as np

import tfrt2.wavelength as wv

path = pathlib.Path("./spd_files")
output_dir = pathlib.Path(".")
spd_paths = path.glob("*.spd")

for path in spd_paths:
    output_path = output_dir / pathlib.Path(path.stem).with_suffix(".dat")

    with open(path, "r") as in_file:
        # I am not certain about this, but it seems like I have to interpret the file as end, start rather than
        # start, end.  But this needs to be checked.
        end, start, _ = in_file.readline().split()
        start, end = float(start), float(end)
        print(f"start: {start}, end: {end}")
        power = np.array(in_file.readlines(), dtype=float)
        point_count = len(power)
        wavelengths = np.linspace(start, end, point_count)

        s = wv.Spectrum((wavelengths, power), limits=(start, end), interp_res=point_count)
        s.save(output_path)
