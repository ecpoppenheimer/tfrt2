import tfrt2.src.optical_system as optical_system
import tfrt2.src.sources as sources
import tfrt2.src.distributions as distributions
import tfrt2.src.wavelength as wavelength
import tfrt2.src.materials as materials


def get_system(driver):
    return LocalSystem(driver)


class LocalSystem(optical_system.OpticalSystem):
    def __init__(self, driver):
        super().__init__(
            driver,
            {
                "source1": "source",
                "source2": "source"
            },
            [materials.vacuum]
        )

        self.feed_parts(
            source1=sources.Source3D(
                self.driver,
                self.self_path,
                self.settings.source1,
                "",
                #base_points=distributions.PixelatedCircle(self.settings.source1, resolution=51),
                base_points=distributions.PerfectCircle(self.settings.source1, driver, resolution=51, mode="base_points"),
                #base_points=distributions.Square(self.settings.source1),
                aperture=distributions.Square(self.settings.source1, driver, mode="aperture"),
                #angles=distributions.PerfectUniformSphere(self.settings.source1)
                #angles=distributions.PerfectLambertianSphere(self.settings.source1)
            ),
            source2=sources.Source3D(
                self.driver,
                self.self_path,
                self.settings.source2,
                wavelength.YELLOW,
                # base_points=distributions.PixelatedCircle(self.settings.source2, resolution=51),
                # base_points=distributions.PerfectCircle(self.settings.source2, resolution=51, mode="base_points"),
                # base_points=distributions.Square(self.settings.source2),
                # aperture=distributions.Square(self.settings.source2, mode="aperture"),
                # angles=distributions.PerfectUniformSphere(self.settings.source2)
                # angles=distributions.PixelatedLambertianSphere(self.settings.source2, resolution=51)
                angles=distributions.PerfectLambertianSphere(self.settings.source2, driver)
            )
        )
