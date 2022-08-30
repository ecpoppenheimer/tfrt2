from math import pi

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

        entrance_max_height = 2
        exit_min_height = 2.2
        radius = 1
        target_edge_length = .05

        zero_points = mt.circular_mesh(radius, target_edge_length)
        print(
            f"zero points has {zero_points.points.shape[0]} vertices and {zero_points.faces.shape[0]/4} faces."
        )

        def filter_quad_symmetry(vertices):
            x, y = vertices[:, 0], vertices[:, 1]
            theta = np.arctan2(y, x)
            return np.logical_and(theta >= 0, theta <= pi/1.99)

        def close(a, b, val=.001):
            return np.abs(np.abs(a) - np.abs(b)) < val

        def attach_quad_symmetry(vertex, available_vertices):
            # want to match (x, y) to...  (-x, y), (x, -y), (-x, -y)
            vx, vy = vertex[0], vertex[1]
            avail_x, avail_y = available_vertices[:, 0], available_vertices[:, 1]
            return np.logical_and(close(vx, avail_x), close(vy, avail_y))

        entrance = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.entrance,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=zero_points,
            mat_in=1,
            mat_out=0,
            filter_drivers=filter_quad_symmetry,
            attach_to_driver=attach_quad_symmetry,
            constraints=[
                optics.ThicknessConstraint(self.settings.entrance, entrance_max_height, False),
            ],
            flip_norm=True
        )

        exit = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.exit,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=zero_points,
            mat_in=1,
            mat_out=0,
            filter_drivers=filter_quad_symmetry,
            attach_to_driver=attach_quad_symmetry,
            constraints=[
                optics.ThicknessConstraint(self.settings.exit, exit_min_height, True),
            ],
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, "",
            base_points=distributions.Square(self.settings.source, driver, x_width=.1, y_width=1),
            aperture=distributions.PerfectCircle(self.settings.source, driver, radius, mode="aperture"),
            aperture_distance=(entrance_max_height + exit_min_height) / 2
        )

        self.feed_parts(source=source, entrance=entrance, exit=exit)

        self.set_goal(CPlaneGoal(
            self.driver, self.settings, "uniform", ('x', -1.0, 1.0), ('y', -1.0, 1.0), 12,
        ))
