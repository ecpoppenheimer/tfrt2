import tfrt2.src.optical_system as optical_system
import tfrt2.src.sources as sources
import tfrt2.src.distributions as distributions
import tfrt2.src.wavelength as wavelength


def get_system(client):
    return LocalSystem(client)


class LocalSystem(optical_system.OpticalSystem):
    def __init__(self, client):
        super().__init__(
            client,
            {
                "source1": "source",
                "source2": "source"
            }
        )

        self.feed_parts({
            "source1": sources.Source3D(
                self.settings.source1,
                "",
                self.self_path,
                #base_points=distributions.PixelatedCircle(self.settings.source1, resolution=51),
                base_points=distributions.PerfectCircle(self.settings.source1, resolution=51, mode="base_points"),
                #base_points=distributions.Square(self.settings.source1),
                aperture=distributions.Square(self.settings.source1, mode="aperture"),
                #angles=distributions.PerfectUniformSphere(self.settings.source1)
                #angles=distributions.PerfectLambertianSphere(self.settings.source1)
            ),
            "source2": sources.Source3D(
                self.settings.source2,
                wavelength.YELLOW,
                self.self_path,
                # base_points=distributions.PixelatedCircle(self.settings.source2, resolution=51),
                # base_points=distributions.PerfectCircle(self.settings.source2, resolution=51, mode="base_points"),
                # base_points=distributions.Square(self.settings.source2),
                # aperture=distributions.Square(self.settings.source2, mode="aperture"),
                # angles=distributions.PerfectUniformSphere(self.settings.source2)
                # angles=distributions.PixelatedLambertianSphere(self.settings.source2, resolution=51)
                angles=distributions.PerfectLambertianSphere(self.settings.source2)
            )
        })
