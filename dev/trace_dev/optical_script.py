import numpy as np
import pyvista as pv
import pathlib

import tfrt2.src.mesh_tools as mt
import tfrt2.src.materials as materials
import tfrt2.src.optics as optics
import tfrt2.src.vector_generator as vg
import tfrt2.src.sources as sources
import tfrt2.src.distributions as distributions
from tfrt2.src.optical_system import OpticalSystem


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "lens1": "optical",
                "lens2": "optical",
                "target": "target",
                "stop": "stop",
                "y_source": "source",
                "x_source": "source"
            },
            [materials.vacuum, materials.acrylic]
        )

        self.settings.lens1.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "test_output.stl"),
        )
        self.settings.lens2.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(self.self_path) / "test_output2.stl"),
        )
        self.settings.target.establish_defaults(
            visible=False,
            color="grey",
            show_edges=False,
        )
        self.settings.stop.establish_defaults(
            visible=False,
            color="red",
            show_edges=False,
        )

        zero_points = mt.circular_mesh(1, .6)
        lens1 = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.lens1,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=zero_points,
            mat_in=1,
            mat_out=0,
            #filter_fixed=filter_fixed,
            flip_norm=True,
            constraints=[optics.SpacingConstraint(self.settings.lens1, .01)]
        )
        self.feed_part("lens1", lens1)
        lens2 = optics.ParametricTriangleOptic(
            self.driver,
            self.self_path,
            self.settings.lens2,
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            mesh=zero_points,
            mat_in=1,
            mat_out=0,
            # filter_fixed=filter_fixed,
            constraints=[optics.SpacingConstraint(self.settings.lens1, .1, target=lens1)]
        )
        self.feed_part("lens2", lens2)

        stop = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.stop,
            mesh=pv.Disc((0, 0, 0), 1.1, 50, c_res=6).triangulate()
        )
        target = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.target,
            mesh=pv.Plane((0, 0, 3), (0, 0, 1), 5, 5, 1, 1).triangulate()
        )
        x_source = sources.Source3D(
            self.driver, self.self_path, self.settings.x_source, "",
            angles=distributions.PerfectUniformSphere(self.settings.x_source, driver)
        )
        y_source = sources.Source3D(
            self.driver, self.self_path, self.settings.y_source, 550,
            angles=distributions.PerfectUniformSphere(self.settings.y_source, driver)
        )

        self.feed_parts(stop=stop, target=target, x_source=x_source, y_source=y_source)

        #self.fuse()
