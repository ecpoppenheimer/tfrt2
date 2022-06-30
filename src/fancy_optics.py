"""
Define base classes that the client is able to recognize.  Individual scripts should inherit from these classes
to be used with the client.
"""
import traceback

import numpy as np
from scipy.interpolate import interp1d
import pyvista as pv
import tensorflow as tf

import tfrt.boundaries as boundaries
import tfrt.mesh_tools as mt
import tfrt.engine as engine
import tfrt2.src.settings as settings


class LinearOptic(boundaries.MasterSlaveParametricTriangleBoundary):
    def __init__(
        self,
        minimum_extent,
        initial_parameter_count,
        initial_zero_points_path,
        optic_settings
    ):
        self.settings = optic_settings
        self.settings.establish_defaults(
            minimum_extent=minimum_extent,
            parameter_count=initial_parameter_count,
            zero_points_input_path=initial_zero_points_path
        )

        self.accumulator = np.eye(self.settings.parameter_count)
        self.masters = tuple()
        self.x_extent = (-1.0, 1.0)
        self.maximum_extent = 1.0
        self.y_extent = (-1.0, 1.0)
        self.initial_profile = lambda x: np.zeros_like(x)
        self.zero_mesh = None
        self.initials = None
        vg = boundaries.FromVectorVG((0.0, 0.0, 1.0))
        self.vertex_update_map = None
        self._parameters = None

        self.build_from_zero_points()
        super().__init__(
            self.zero_mesh,
            vg,
            auto_update_mesh=True,
            material_dict={"mat_in": 1, "mat_out": 0},
            initial_parameters=self.initials,
            validate_shape=False
        )

    @staticmethod
    def extract_profile_from_mesh(points):
        points = points[points[:, 0] > 0]
        y = points[:, 1]
        z = points[:, 2]

        # Sort the points in increasing order of the y coordinate
        order = np.argsort(y)
        y = y[order]
        z = z[order]

        return interp1d(y, z)

    def extract_profile_from_parameters(self):
        points = self.zero_mesh.points
        points = points[points[:, 0] > 0]
        y = points[:, 1]
        z = self.parameters
        return interp1d(y, z)

    def find_extents(self):
        starting_points = pv.read(self.settings.zero_points_input_path).points
        self.x_extent = np.amin(np.array(starting_points[:, 0])), np.amax(np.array(starting_points[:, 0]))
        self.maximum_extent = np.amax(starting_points[:, 2])
        self.y_extent = np.amin(np.array(starting_points[:, 1])), np.amax(np.array(starting_points[:, 1]))
        self.initial_profile = self.extract_profile_from_mesh(starting_points)

    def zero_points_from_profile(self, profile, parameter_count):
        ys = np.linspace(*self.y_extent, parameter_count)
        zs = profile(ys)
        points = [(self.x_extent[0], ys[0], 0.0), (self.x_extent[1], ys[0], 0.0)]
        faces = []
        point_count = 2

        for y, z in zip(ys[1:], zs[1:]):
            points.append((self.x_extent[0], y, 0))
            points.append((self.x_extent[1], y, 0))
            point_count += 2
            faces.append((point_count - 4, point_count - 2, point_count - 1))
            faces.append((point_count - 1, point_count - 3, point_count - 4))

        return pv.PolyData(np.array(points), mt.pack_faces(faces)), zs

    def remesh(self):
        self.zero_mesh, self.initials = self.zero_points_from_profile(
            self.extract_profile_from_parameters(), self.settings.parameter_count
        )
        self._zero_points = self.zero_mesh.points.copy()
        print("remesh, just made _zero_points")
        self.reparametrize()
        print("remesh, just did reparametrize")
        self.build_parameters()
        print("remesh, just did build parameters")
        self._vertices = tf.constant(self._zero_points, dtype=tf.float64)
        self._faces = tf.reshape(self._mesh.faces, (-1, 4))

        # Pick the top vertex as the master vertex in the middle-most position among the masters
        self.masters = self.filter_masters(self._zero_points)
        print(f"master shape: {self.masters.shape}")
        top_v = self.masters[np.floor(self.masters.shape[0] / 2).astype(int)]
        print(f"zero mesh v shape: {self.zero_mesh.points.shape}")
        print(f"top v: {top_v}")

        self.vertex_update_map, self.accumulator = mt.mesh_parametrization_tools(self.zero_mesh, top_v, self.masters)
        print("remesh, just did mesh parametrizeation tools")
        print(f"vup shape: {self.vertex_update_map.shape}")

    def build_from_zero_points(self):
        self.find_extents()
        self.zero_mesh, self.initials = self.zero_points_from_profile(
            self.initial_profile, self.settings.parameter_count
        )

        # Pick the top vertex as the master vertex in the middle-most position among the masters
        self.masters = self.filter_masters(self.zero_mesh.points)
        top_v = self.masters[np.floor(self.masters.shape[0] / 2).astype(int)]

        self.vertex_update_map, self.accumulator = mt.mesh_parametrization_tools(self.zero_mesh, top_v, self.masters)

    def save(self, filepath):
        if self._mesh:
            self._mesh.save(filepath)


class FancySystem(engine.OpticalSystem3D):
    def __init__(self, system_path, verbose=False):
        """
        A wrapper for a tfrt.engine.OpticalSystem that provides a more friendly interface for the client.

        If system_path is None, then this system will not synchronize its settings with a file, and the values
        used will be the values defined in the script.  If a path is given to system_path, then the system
        will try to synchronize its settings with the file at that path.  On close, it will save the values of
        its settings to this file, and on instantiation it will overwrite its settings with the ones in this file.
        Thus, be aware that if system_path is specified, the values defined in code act only as initial values.  No
        method of recovering initial values is currently implemented, but this can be done system-wide by deleting
        the settings file.  The file synchronization process is designed to gracefully handle errors.  By which I mean
        it will ignore errors, often without a warning, and resort to defaults, unless the verbose is set to true,
        in which case errors will be printed to screen.

        Parameters
        ----------
        parts : dict of the format <str>: (<str>, <value>)
            A dictionary containing the parts that will go into the optical system.  Keys are the name of each
            part, which can be used to access that part later in the program.  Values are 2-tuples, where the first
            element is a string that labels the type of the part which must be one of {optical, stop, target, source},
            and the second element is the part itself.
        system_path : str or pathlib.Path
            The path to a settings.dat file to synchronize component settings.
        verbose : bool, optional
            If False, the default, various recoverable errors will be handled without printing an error message.
            If True, error messages will always be displayed.
        """
        self.parts = {}
        self.system_path = None
        super().__init__()
        self.verbose = verbose

        # Load the system settings for this system
        self.system_path = system_path
        self.settings_path = system_path / "settings.data"
        self.settings = settings.Settings()
        try:
            self.settings.load(self.settings_path)
        except Exception:
            if self.verbose:
                print("FancySystem: got exception while trying to load the system settings:")
                print(traceback.format_exc())

        self.optical_parts = []
        self.stop_parts = []
        self.target_parts = []
        self.source_parts = []
        self._optical_boundaries = []

        self.optical = []
        self.stops = []
        self.targets = []
        self.sources = []

    def build_from_parts(self, parts):
        for key, tpl in parts.items():
            category, component = tpl
            component.system_path = self.system_path

            # Avoid duplicate names: names must be unique
            if key in set(self.parts.keys()):
                raise ValueError(f"FancySystem: name {key} is already in use for this system.  Names must be unique.")
            component.name = key

            # Place components into the correct bucket by category
            if category == "optical":
                self.optical_parts.append(component)
                try:
                    self._optical_boundaries.append(component.boundary)
                except AttributeError:
                    self._optical_boundaries.append(component)
            elif category == "stop":
                self.stop_parts.append(component)
            elif category == "target":
                self.target_parts.append(component)
            elif category == "source":
                self.source_parts.append(component)
            else:
                raise ValueError(f"FancySystem: category {category} not in {{optical, stop, target, source}}")
            self.parts[key] = component

        self.optical = self._optical_boundaries
        self.stops = self.stop_parts
        self.targets = self.target_parts
        self.sources = self.source_parts

    def update(self):
        for component in self.parts.values():
            component.update()

    def save(self):
        if self.settings_path is not None:
            for key, component in self.parts.items():
                if hasattr(component, "settings"):
                    self.settings.dict[key] = component.settings.dict
            self.settings.save(self.settings_path)
