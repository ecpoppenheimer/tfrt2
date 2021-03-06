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
                "prism": "optical",
                "target": "target",
                "source": "source",
            },
            [materials.vacuum, materials.acrylic]
        )

        self.settings.prism.establish_defaults(
            visible=True,
            color="cyan",
            show_edges=False,
            mesh_output_path=str(pathlib.Path(self.self_path) / "test_output.stl"),
            mesh_input_path=str(pathlib.Path(self.self_path) / "test_input.stl"),
        )
        self.settings.target.establish_defaults(
            visible=False,
            color="grey",
            show_edges=False,
        )

        prism = optics.TriangleOptic(
            self.driver,
            self.self_path,
            self.settings.prism,
            mesh=pv.Tetrahedron().triangulate(),
            mat_in=1,
            mat_out=0,
            flip_norm=True
        )

        target = optics.TriangleOptic(
            self.driver, self.self_path, self.settings.target,
            mesh=pv.Plane((-10, 0, 0), (1, 0, 0), 50, 50, 1, 1).triangulate()
        )
        source = sources.Source3D(
            self.driver, self.self_path, self.settings.source, "",
            angles=distributions.PerfectUniformSphere(self.settings.source, driver)
        )

        self.feed_parts(target=target, source=source, prism=prism)
