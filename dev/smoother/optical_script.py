import numpy as np
import pyvista as pv
import pathlib

import tfrt2.mesh_tools as mt
import tfrt2.materials as materials
import tfrt2.optics as optics
import tfrt2.vector_generator as vg
import tfrt2.sources as sources
import tfrt2.distributions as distributions
from tfrt2.optical_system import OpticalSystem
from tfrt2.goal import CPlaneGoal


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "lens": "optical",
            },
            [materials.vacuum, materials.acrylic]
        )

        self.settings.lens.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
        )

        mesh = pv.PolyData(mt.circular_mesh(1, .5))
        print(f"making lens with vertex count {mesh.points.shape[0]}")
        lens = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.lens,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=mesh,
            mat_in=1,
            mat_out=0,
            flip_norm=True,
        )
        self.feed_part("lens", lens)
