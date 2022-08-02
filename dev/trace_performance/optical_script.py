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


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "entrance": "optical",
                "lens": "optical",
                "target": "target",
                "source": "source",
            },
            [materials.vacuum, materials.acrylic]
        )

        self.settings.entrance.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=False,
            mesh_output_path=str(pathlib.Path(self.self_path) / "entrance_output.stl"),
            mesh_input_path=str(pathlib.Path(self.self_path) / "mesh_input.stl"),
        )
        self.settings.lens.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "lens_output.stl"),
        )
        self.settings.target.establish_defaults(
            visible=False,
            color="grey",
            show_edges=False,
        )

        entrance_height = .005
        radius = .1
        target_distance = 20

        entrance = optics.TriangleOptic(
            self.driver,
            self.self_path,
            self.settings.entrance,
            mat_in=1,
            mat_out=0,
            flip_norm=True,
        )
        entrance.update()
        self.feed_part("entrance", entrance)

        def filter_fixed(vertices):
            return np.sum((vertices**2)[:, :-1], axis=1) > (.98*radius) ** 2

        lens = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.lens,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=mt.circular_mesh(radius, .01).translate((0, 0, entrance_height), inplace=True),
            mat_in=1,
            mat_out=0,
            filter_fixed=filter_fixed,
            constraints=[optics.ClipConstraint(self.settings.lens, entrance_height + .05, 100)]
        )
        self.feed_part("lens", lens)

        target = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.target,
            mesh=pv.Plane((0, 0, target_distance), (0, 0, 1), 10000, 10000, 1, 1).triangulate()
        )

        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, 550,
            base_points=distributions.Square(self.settings.source, driver, x_width=.05, y_width=.05),
            angles=distributions.PerfectLambertianSphere(self.settings.source, driver)
        )

        self.feed_parts(target=target, source=source)

        print("Built optical script!")
