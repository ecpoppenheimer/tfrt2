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
                "entrance": "optical",
                "exit": "optical",
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
        self.settings.exit.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "exit_output.stl"),
        )

        entrance_min_height = .005
        entrance_max_height = .1
        entrance_radius = .1

        def filter_fixed_entrance(vertices):
            return np.sum((vertices**2)[:, :-1], axis=1) > (.98*entrance_radius) ** 2

        entrance = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.entrance,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=mt.circular_mesh(entrance_radius, .015),
            mat_in=1,
            mat_out=0,
            filter_fixed=filter_fixed_entrance,
            constraints=[
                optics.ThicknessConstraint(self.settings.entrance, entrance_min_height, True),
                optics.ClipConstraint(self.settings.entrance, entrance_min_height, entrance_max_height)
            ],
            flip_norm=True
        )
        print(f"built entrance with {entrance.vertices.shape[0]} vertices and {entrance.faces.shape[0]} faces.")

        exit_min_height = .15
        exit_max_height = .6
        exit_fixed_height = .25
        exit_radius = .2

        def filter_fixed_exit(vertices):
            return np.sum((vertices ** 2)[:, :-1], axis=1) > (.98 * exit_radius) ** 2

        exit = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.exit,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=mt.circular_mesh(exit_radius, .03),
            mat_in=1,
            mat_out=0,
            filter_fixed=filter_fixed_exit,
            constraints=[
                optics.PointConstraint(self.settings.exit, exit_fixed_height, (0, 0, 0)),
                optics.ClipConstraint(self.settings.exit, exit_min_height, exit_max_height)
            ]
        )
        print(f"built exit with {exit.vertices.shape[0]} vertices and {exit.faces.shape[0]} faces.")

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, "",
            base_points=distributions.Square(self.settings.source, driver, x_width=.1, y_width=1),
            angles=distributions.PerfectLambertianSphere(self.settings.source, driver)
        )

        self.feed_parts(source=source, entrance=entrance, exit=exit)

        self.set_goal(CPlaneGoal(
            self.driver, self.settings, "uniform", ('x', -1.0, 1.0), ('y', -1.0, 1.0), 5,
        ))
