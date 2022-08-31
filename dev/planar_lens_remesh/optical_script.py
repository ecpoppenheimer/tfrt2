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

        minimum_spacing = .1

        square_zero_points = pv.Plane().triangulate()
        circular_zero_points = mt.circular_mesh(1, .1)

        def close(a, b, val=.001):
            return np.abs(np.abs(a) - np.abs(b)) < val

        def fix_square_edges(vertices):
            x, y = vertices[:, 0], vertices[:, 1]
            x_min, x_max = np.amin(x), np.amax(x)
            y_min, y_max = np.amin(y), np.amax(y)

            x_edges = np.logical_or(close(x, x_min), close(x, x_max))
            y_edges = np.logical_or(close(y, y_min), close(y, y_max))
            return np.logical_or(x_edges, y_edges)

        def fix_circle_edges(vertices):
            x, y = vertices[:, 0], vertices[:, 1]
            return close(x ** 2 + y ** 2, 1)

        def filter_quad_symmetry(vertices):
            x, y = vertices[:, 0], vertices[:, 1]
            theta = np.arctan2(y, x)
            return np.logical_and(theta >= 0, theta <= pi/1.99)

        def attach_quad_symmetry(vertex, available_vertices):
            # want to match (x, y) to...  (-x, y), (x, -y), (-x, -y)
            vx, vy = vertex[0], vertex[1]
            avail_x, avail_y = available_vertices[:, 0], available_vertices[:, 1]
            return np.logical_and(close(vx, avail_x), close(vy, avail_y))

        entrance = optics.ReparameterizablePlanarOptic(
            self.driver,
            self.self_path,
            self.settings.entrance,
            "xy",
            "square",
            mesh=square_zero_points,
            mat_in=1,
            mat_out=0,
            filter_fixed=fix_square_edges,
            filter_drivers=filter_quad_symmetry,
            attach_to_driver=attach_quad_symmetry,
            constraints=[
                optics.ThicknessConstraint(self.settings.entrance, 0, False),
            ],
            flip_norm=False
        )

        exit = optics.ReparameterizablePlanarOptic(
            self.driver,
            self.self_path,
            self.settings.exit,
            "xy",
            "circle",
            mesh=circular_zero_points,
            mat_in=1,
            mat_out=0,
            filter_fixed=fix_circle_edges,
            filter_drivers=filter_quad_symmetry,
            attach_to_driver=attach_quad_symmetry,
            constraints=[
                optics.ThicknessConstraint(self.settings.exit, minimum_spacing, True),
            ],
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, "",
            base_points=distributions.Square(self.settings.source, driver, x_width=.1, y_width=1),
            angles=distributions.PerfectUniformSphere(self.settings.source, driver, angular_cutoff=30),
        )

        self.feed_parts(source=source, entrance=entrance, exit=exit)

        self.set_goal(CPlaneGoal(
            self.driver, self.settings, "uniform", ('x', -1.0, 1.0), ('y', -1.0, 1.0), 12,
        ))
