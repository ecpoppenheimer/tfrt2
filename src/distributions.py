import math

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import numpy as np

import cumdistf
import tfrt2.src.component_widgets as cw


class Square(qtw.QWidget):
    """
    Generate 2D points uniformly distributed in a square (actually a rectangle), given a 2D uniform seed.
    """
    def __init__(
            self, settings, driver, x_width=1.0, y_width=1.0, mode="base_points"
    ):
        if mode == "base_points":
            x_key, y_key = "bp_x_radius", "bp_y_radius"
            label = "Full width of the base points"
        elif mode == "aperture":
            x_key, y_key = "ap_x_radius", "ap_y_radius"
            label = "Full width of the aperture"
        else:
            raise RuntimeError("Square: Mode must be either 'base_points' or 'aperture'.")
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(**{x_key: x_width/2, y_key: y_width/2})
        self._x_key = x_key
        self._y_key = y_key
        layout = qtw.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(qtw.QLabel(label), 0, 0, 1, 2)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, x_key, float, validator=qtg.QDoubleValidator(0.0, 100000, 6), label="x",
            callback=driver.try_auto_retrace
        ), 1, 0, 1, 1)
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, y_key, float, validator=qtg.QDoubleValidator(0.0, 100000, 6), label="y",
            callback=driver.try_auto_retrace
        ), 1, 1, 1, 1)

    def __call__(self, seed):
        return (seed - .5) * ((self.settings.dict[self._x_key], self.settings.dict[self._y_key]),)


class PixelatedCircle(qtw.QWidget):
    """
    Generate 2D points uniformly distributed in a circle, given a 2D uniform seed.

    This class uses a CumulativeDistributionFunction to convert from a square seed into a circular output.  This
    implementation is less efficient than PerfectCircle, but generates a direct mapping from the square seed to
    the circular output, meaning the seed could be used for goal definition with a square goal.
    """
    def __init__(self, settings, driver, radius=1.0, resolution=50, mode="base_points"):
        if mode == "base_points":
            r_key = "bp_radius"
            label = "Radius of the base points"
        elif mode == "aperture":
            r_key = "ap_radius"
            label = "Radius of the aperture"
        else:
            raise RuntimeError("PixelatedCircle: Mode must be either 'base_points' or 'aperture'.")
        super().__init__()
        if resolution % 2 == 0:
            resolution += 1  # doesn't work if resolution isn't odd.
        self.settings = settings
        self._r_key = r_key
        self.settings.establish_defaults(**{r_key: radius})
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(cw.SettingsEntryBox(
            self.settings, r_key, float, validator=qtg.QDoubleValidator(0.0, 100000, 6),
            label=label, callback=driver.try_auto_retrace
        ))

        x = np.linspace(-.5, .5, resolution)
        density = x[:, None]**2 + x[None, :]**2 <= .25
        self._cdf = cumdistf.CumulativeDistributionFunction2D(
            ((-.5, .5), (-.5, .5)), density=density, direction="forward", dtype=np.float64
        )

    def __call__(self, seed):
        return self._cdf(seed) * 2 * self.settings.dict[self._r_key]


class PerfectCircle(qtw.QWidget):
    """
    Generate 2D points uniformly distributed in a circle, given a 2D uniform seed.

    This class uses an analytic CDF which is more efficient than PixelatedCircle and produces an exact circular output.
    However, it mangles the meaning of its seed: It uses one dimension as a radius and the other as an angle and uses
    a Fibonacci Spiral to make its points.
    """

    def __init__(self, settings, driver, radius=1.0, resolution=50, mode="base_points"):
        if mode == "base_points":
            r_key = "bp_radius"
            label = "Radius of the base points"
        elif mode == "aperture":
            r_key = "ap_radius"
            label = "Radius of the aperture"
        else:
            raise RuntimeError("PerfectCircle: Mode must be either 'base_points' or 'aperture'.")
        super().__init__()
        if resolution % 2 == 0:
            resolution += 1  # doesn't work if resolution isn't odd.
        self.settings = settings
        self._r_key = r_key
        self._factor = 2 * math.pi * (1 + 5**0.5)
        self.settings.establish_defaults(**{r_key: radius})
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(cw.SettingsEntryBox(
            self.settings, r_key, float, validator=qtg.QDoubleValidator(0.0, 100000, 6),
            label=label, callback=driver.try_auto_retrace
        ))

    def __call__(self, seed):
        r = self.settings.dict[self._r_key] * np.sqrt(seed[:, 0]).reshape((-1, 1))
        theta = self._factor * seed[:, 1]
        return r * np.stack((np.cos(theta), np.sin(theta)), axis=1)


class PerfectUniformSphere(qtw.QWidget):
    """
    Generate 3D points uniformly distributed on the surface of a sphere, given a 2D uniform seed.

    This class uses an analytic CDF which is more efficient than a cumdistf distribution and produces an exact spherical
    output.  However, it mangles the meaning of its seed: It interprets the seed as spherical coordinates, and uses
    one dimension as phi and the other as theta.
    """

    def __init__(self, settings, driver, ray_length=1.0, angular_cutoff=90.0):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(ray_length=ray_length, angular_cutoff=angular_cutoff)
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "ray_length", float, validator=qtg.QDoubleValidator(0.0, 100000, 6),
            callback=driver.try_auto_retrace
        ))
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "angular_cutoff", float, validator=qtg.QDoubleValidator(0.0, 180.0, 6),
            label="Angular Cutoff (degrees)", callback=driver.try_auto_retrace
        ))

    def __call__(self, seed):
        # Phi will go from 1 to cos(angular_cutoff), and the proper cdf is just arccos.
        phi = shift(seed[:, 0], 0.0, 1.0, math.cos(math.radians(self.settings.angular_cutoff)), 1)
        # phi = np.arccos(phi)

        # Old implementation uses phi, and take the arccos, followed by taking a cos and a sin in the stack.  But
        # it will probably be more efficient to simplify.
        sin_arccos_phi = np.sqrt(1 - phi**2)

        # older implementation had a factor of the golden ratio in here, but I am just not seeing what good that
        # would do, except possibly in the unusual circumstance of using a non-random seed.  If the user wants to
        # use a non-random seed, they can add in their own golden ratio.
        theta = 2 * math.pi * seed[:, 1]

        """return self.settings.ray_length * np.stack((
            np.cos(phi),
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta)
        ), axis=1)"""
        return self.settings.ray_length * np.stack((
            sin_arccos_phi * np.cos(theta),
            sin_arccos_phi * np.sin(theta),
            phi
        ), axis=1)


class PerfectLambertianSphere(qtw.QWidget):
    """
    Generate 3D points on the surface of a sphere in a Lambertian (cosine) distribution, given a 2D uniform seed.

    This class uses an analytic CDF which is more efficient than a cumdistf distribution and produces an exact spherical
    output.  However, it mangles the meaning of its seed: It interprets the seed as spherical coordinates, and uses
    one dimension as phi and the other as theta.
    """

    def __init__(self, settings, driver, ray_length=1.0, angular_cutoff=90.0):
        super().__init__()
        self.settings = settings
        self.settings.establish_defaults(ray_length=ray_length, angular_cutoff=angular_cutoff)
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "ray_length", float, validator=qtg.QDoubleValidator(0.0, 100000, 6),
            callback=driver.try_auto_retrace
        ))
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "angular_cutoff", float, validator=qtg.QDoubleValidator(0.0, 90.0, 6),
            label="Angular Cutoff (degrees)", callback=driver.try_auto_retrace
        ))

    def __call__(self, seed):
        # Phi will go from 1 to cos(angular_cutoff).
        phi = shift(seed[:, 0], 0.0, 1.0, math.cos(math.radians(self.settings.angular_cutoff)), 1)
        # phi = np.arccos(phi)

        # Analytic CDF for this one differs from the uniform sphere only in that we need to take the sqrt of phi first.
        # But this is simplifiable!  This also motivates how phi can only go between 0 and 90 for this case.  But
        # how often are you going to want more from a Lambertian source?
        # phi = arccos(sqrt(phi))
        # rtn = stack(
        #    sin(phi) * np.cos(theta),
        #    sin(phi) * np.sin(theta),
        #    cos(phi)
        #)
        sin_arccos_phi = np.sqrt(1 - phi)

        # older implementation had a factor of the golden ratio in here, but I am just not seeing what good that
        # would do, except possibly in the unusual circumstance of using a non-random seed.  If the user wants to
        # use a non-random seed, they can add in their own golden ratio.
        theta = 2 * math.pi * seed[:, 1]

        """return self.settings.ray_length * np.stack((
            np.cos(phi),
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta)
        ), axis=1)"""
        return self.settings.ray_length * np.stack((
            sin_arccos_phi * np.cos(theta),
            sin_arccos_phi * np.sin(theta),
            np.sqrt(phi)
        ), axis=1)


class PixelatedLambertianSphere(qtw.QWidget):
    """
    Generate 3D points on the surface of a sphere in a Lambertian (cosine) distribution, given a 2D uniform seed.

    This class uses a CumulativeDistributionFunction to convert from a square seed into a circular one.  This
    implementation is less efficient than PerfectLambertianSphere, but generates a direct mapping from the square seed
    to the spherical output, meaning the seed could be used for goal definition with a square goal.
    """

    def __init__(self, settings, driver, ray_length=1.0, resolution=51, angular_cutoff=90.0):
        super().__init__()
        if resolution % 2 == 0:
            resolution += 1  # doesn't work if resolution isn't odd.
        self._resolution = resolution
        self.settings = settings
        self.settings.establish_defaults(ray_length=ray_length, angular_cutoff=angular_cutoff)
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "ray_length", float, validator=qtg.QDoubleValidator(0.0, 100000, 6),
            callback=driver.try_auto_retrace
        ))
        layout.addWidget(cw.SettingsEntryBox(
            self.settings, "angular_cutoff", float, validator=qtg.QDoubleValidator(0.0, 90.0, 6),
            label="Angular Cutoff (degrees)", callback=[self._refresh_cdf, driver.try_auto_retrace]
        ))

        self._cdf = None
        self._refresh_cdf()

    def _refresh_cdf(self):
        # The density is a circle centered on the origin with radius = sin(cutoff), which will be 1 when cutoff=90,
        # so density will always be within the unit square.
        x = np.linspace(-1.0, 1.0, self._resolution)
        density = x[:, None] ** 2 + x[None, :] ** 2 <= math.sin(math.radians(self.settings.angular_cutoff))**2
        self._cdf = cumdistf.CumulativeDistributionFunction2D(
            ((-1.0, 1.0), (-1.0, 1.0)), density=density, direction="forward", dtype=np.float64
        )

    def __call__(self, seed):
        # Generate points on a circle using the CDF.
        circle_x, circle_y = np.moveaxis(self._cdf(seed), 1, 0)

        print(f"circle_x: {circle_x.shape}, min: {np.amin(circle_x)}, max: {np.amax(circle_x)}")
        print(f"circle_y: {circle_y.shape}, min: {np.amin(circle_y)}, max: {np.amax(circle_y)}")

        # Now project onto a sphere centered at z=1 to get spherical coordinates.  This projection automatically
        # produces a lambertian distribution!
        xx = circle_x ** 2
        yy = circle_y ** 2
        circle_rad = np.sqrt(xx + yy)
        zz = np.clip(1 - xx - yy, 0.0, 1.0)

        # Now move back into cartesian coordinates.  Doing trig substitutions.
        # phi = np.arctan2(circle_rad, z)
        #theta = np.arctan2(circle_y, circle_x)
        phi_hyp = np.sqrt(xx + yy + zz)
        sin_phi = circle_rad / phi_hyp
        cos_phi = np.sqrt(zz) / phi_hyp
        return np.stack((
            sin_phi * circle_x / circle_rad,  # np.sin(phi) * np.cos(theta)
            sin_phi * circle_y / circle_rad,  # np.sin(phi) * np.sin(theta),
            cos_phi
        ), axis=1)


# print(f"distribution x min: {np.amin(rtn[:, 0])}, x max: {np.amax(rtn[:, 0])}, y min: {np.amin(rtn[:, 1])}, y max: {np.amax(rtn[:, 1])}")

def shift(pts, x1, x2, y1, y2):
    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    return pts * m + b
