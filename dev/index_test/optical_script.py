import pyvista as pv

import tfrt2.materials as materials
import tfrt2.optics as optics

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
                "light": "optical",
                "heavy": "optical",
                "source": "source",
                "target": "target",
            },
            [materials.vacuum, materials.acrylic, materials.build_constant_material(5)]
        )

        self.settings.light.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=False,
        )
        self.settings.heavy.establish_defaults(
            visible=True,
            color="skyblue",
            show_edges=False,
        )
        self.settings.source.establish_defaults(
            center=[4, 0, 0],
            angle=[-1, 0, 0]
        )
        self.settings.target.establish_defaults(
            visible=False,
            color="grey",
            show_edges=False,
        )

        light = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.light, mat_in=1, mat_out=0,
            mesh=pv.Cylinder((0, -.5, 0), (0, 0, 1), .4, .25, 500).triangulate()
        )
        heavy = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.heavy, mat_in=2, mat_out=0,
            mesh=pv.Cylinder((0, .5, 0), (0, 0, 1), .4, .25, 500).triangulate()
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, 580,
            base_points=distributions.Square(self.settings.source, driver, x_width=2, y_width=.01),
            angles=distributions.PerfectUniformSphere(self.settings.source, driver, angular_cutoff=.1)
        )
        target = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.target,
            mesh=pv.Plane((-5, 0, 0), (1, 0, 0), 10000, 10000, 1, 1).triangulate()
        )

        self.feed_parts(source=source, light=light, heavy=heavy, target=target)
        self.set_goal(CPlaneGoal(
            self.driver, self.settings, "uniform", ('x', -3.0, 3.0), ('y', -3.0, 3.0), 0, ""
        ))
