import numpy as np
import pyvista as pv
import pathlib

import tfrt2.src.settings as settings
import tfrt2.src.materials as materials
import tfrt2.src.optics as optics
import tfrt2.src.vector_generator as vg
from tfrt2.src.optical_system import OpticalSystem


def get_system(client):
    return LocalSystem(client)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "lens": "optical",
            },
            []
        )

        self.settings.lens.establish_defaults(
            visible=True,
            color="green",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "test_output.stl"),
            mesh_input_path=str(pathlib.Path(self.self_path) / "test_input.stl"),
        )

        PARALLEL_AXIS = 1
        PERPENDICULAR_AXIS = 0

        def filter_fixed(vertices):
            s = vertices[:, PARALLEL_AXIS]
            min, max = np.amin(s), np.amax(s)
            return np.logical_or(s > max - .001, s < min + .001)

        def filter_drivers(vertices):
            s = vertices[:, PERPENDICULAR_AXIS]
            min = np.amin(s)
            return s < min + .001

        def attach_to_driver(v, vertices):
            return np.abs(vertices[:, PARALLEL_AXIS] - v[PARALLEL_AXIS]) < .001

        lens = optics.ParametricTriangleOptic(
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            self.client,
            self.self_path,
            self.settings.lens,
            mesh=pv.Plane(direction=(0.0, 0.0, -1.0), i_resolution=5, j_resolution=5).triangulate(),
            mat_in=1,
            mat_out=0,
            filter_fixed=filter_fixed,
            filter_drivers=filter_drivers,
            attach_to_driver=attach_to_driver
        )
        self.feed_part("lens", lens)
