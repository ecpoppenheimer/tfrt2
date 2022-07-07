import numpy as np
import tensorflow as tf
import pyvista as pv
import pathlib

import tfrt2.src.fancy_optics as fancy_optics
import tfrt2.src.settings as settings
import tfrt.materials as materials
import tfrt2.src.sources as sources
import tfrt2.src.distributions as distributions
import tfrt2.src.wavelength as wavelength
import tfrt2.src.component_widgets as cpw
import tfrt2.src.optics as optics
import tfrt2.src.vector_generator as vg


def get_system(client, path_to_self):
    return Mk5System(client, path_to_self)


class Mk5System(fancy_optics.FancySystem):
    def __init__(self, client, path_to_self):
        super().__init__(path_to_self)
        self.client = client

        self.settings.establish_defaults(
            source1=settings.Settings(),
            source2=settings.Settings()
        )

        # Make the source
        source1 = sources.Source3D(
            self.settings.source1,
            "",
            path_to_self,
            #base_points=distributions.PixelatedCircle(self.settings.source1, resolution=51),
            base_points=distributions.PerfectCircle(self.settings.source1, resolution=51, mode="base_points"),
            #base_points=distributions.Square(self.settings.source1),
            aperture=distributions.Square(self.settings.source1, mode="aperture"),
            #angles=distributions.PerfectUniformSphere(self.settings.source1)
            #angles=distributions.PerfectLambertianSphere(self.settings.source1)
        )
        source2 = sources.Source3D(
            self.settings.source2,
            wavelength.YELLOW,
            path_to_self,
            # base_points=distributions.PixelatedCircle(self.settings.source2, resolution=51),
            # base_points=distributions.PerfectCircle(self.settings.source2, resolution=51, mode="base_points"),
            # base_points=distributions.Square(self.settings.source2),
            # aperture=distributions.Square(self.settings.source2, mode="aperture"),
            # angles=distributions.PerfectUniformSphere(self.settings.source2)
            # angles=distributions.PixelatedLambertianSphere(self.settings.source2, resolution=51)
            angles=distributions.PerfectLambertianSphere(self.settings.source2)
        )

        # package the parts
        parts = {
            "source1": ("source", source1),
            "source2": ("source", source2)
        }
        self.build_from_parts(parts)

        self.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
        self.update()
