import numpy as np
import pyvista as pv
import pathlib

import tfrt2.mesh_tools as mt
import tfrt2.materials as materials
import tfrt2.optics as optics
import tfrt2.vector_generator as vg
import tfrt2.sources as sources
import tfrt2.distributions as distributions
import tfrt2.goal as goal
from tfrt2.optical_system import OpticalSystem


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "source": "source",
            },
            [materials.vacuum, materials.acrylic]
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, 580,
            base_points=distributions.Square(self.settings.source, driver, x_width=.05, y_width=.05),
            angles=distributions.PerfectUniformSphere(self.settings.source, driver)
        )

        self.feed_parts(source=source)

        self.set_goal(goal.CPlaneGoal(
            self.driver, self.settings, "cdf", ('x', -3.0, 3.0), ('y', -3.0, 3.0), 0, ""
        ))
