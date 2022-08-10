import numpy as np
import pyvista as pv
import pathlib

import tfrt2.mesh_tools as mt
import tfrt2.materials as materials
import tfrt2.optics as optics
import tfrt2.vector_generator as vg
import tfrt2.sources as sources
import tfrt2.distributions as distributions
import tfrt2.goal as goals
from tfrt2.optical_system import OpticalSystem


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "entrance": "optical",
                "lens": "optical",
                "source": "source",
            },
            [materials.vacuum, materials.acrylic]
        )

        self.settings.entrance.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=False,
            mesh_output_path=str(pathlib.Path(self.self_path) / "entrance_output.stl"),
        )
        self.settings.lens.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "lens_output.stl"),
            mesh_input_path=str(pathlib.Path(self.self_path) / "mesh_input.stl"),
        )

        entrance_height = .005
        radius = .1

        entrance = optics.TriangleOptic(
            self.driver,
            self.self_path,
            self.settings.entrance,
            mesh=pv.Disc((0, 0, entrance_height), inner=0, outer=radius, c_res=32, normal=(0, 0, -1)).triangulate(),
            mat_in=1,
            mat_out=0,
        )
        entrance.update()

        lens = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.lens,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mat_in=1,
            mat_out=0,
            constraints=[optics.ClipConstraint(self.settings.lens, entrance_height + .05, .5)]
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, 580,
            base_points=distributions.Square(self.settings.source, driver, x_width=.05, y_width=.05),
            angles=distributions.PerfectLambertianSphere(self.settings.source, driver)
        )

        self.feed_parts(source=source, lens=lens, entrance=entrance)

        self.set_goal(goals.CPlaneGoal(self.driver, self.settings, "uniform", ('x', -3.0, 3.0), ('y', -3.0, 3.0), 0))
