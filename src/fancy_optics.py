"""
Define base classes that the client is able to recognize.  Individual scripts should inherit from these classes
to be used with the client.
"""

import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
import pyvista as pv

import tfrt.boundaries as boundaries
import tfrt.mesh_tools as mt
import tfrt.engine as engine


class LinearOptic:
    def __init__(
        self,
        initial_filename,
        minimum_extent,
        initial_parameter_count,
        filter_masters,
        attach_slaves
    ):
        self.minimum_extent = minimum_extent
        self.filter_masters = filter_masters
        self.attach_slaves = attach_slaves
        self.acum = np.eye(initial_parameter_count)
        self.masters = tuple()

        # Interpret the initial guess
        starting_points = pv.read(initial_filename).points
        self.x_extent = np.amin(starting_points[:, 0]), np.amax(starting_points[:, 0])
        self.maximum_extent = np.amax(starting_points[:, 2])
        self.y_extent = np.amin(starting_points[:, 1]), np.amax(starting_points[:, 1])
        self.initial_profile = self.extract_profile_from_mesh(starting_points)
        self.initial_parameter_count = initial_parameter_count

        # Make the parametric optic
        self.zero_points, self.initials = self.zero_points_from_profile(
            self.initial_profile, self.initial_parameter_count
        )
        self.boundary = self.build_lens(self.zero_points, self.initials)

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
        points = self.zero_points
        points = points[points[:, 0] > 0]
        y = points[:, 1]
        z = self.boundary.parameters
        return interp1d(y, z)

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

    def remesh(self, parameter_count):
        self.zero_points, self.initials = self.zero_points_from_profile(
            self.extract_profile_from_parameters(), parameter_count
        )
        self.boundary = self.build_lens(self.zero_points, self.initials)

    def build_lens(self, zero_points, initials):
        # Pick the top vertex as the master vertex in the middle-most position among the masters
        self.masters = self.filter_masters(zero_points.points)
        top_v = self.masters[np.floor(self.masters.shape[0] / 2).astype(int)]

        vum, self.acum = mt.mesh_parametrization_tools(zero_points, top_v, self.masters)

        vg = boundaries.FromVectorVG((0.0, 0.0, 1.0))
        lens = boundaries.MasterSlaveParametricTriangleBoundary(
            self.masters,
            self.attach_slaves,
            zero_points,
            vg,
            auto_update_mesh=True,
            material_dict={"mat_in": 1, "mat_out": 0},
            initial_parameters=initials,
            vertex_update_map=vum
        )

        return lens

    def add_constraint(self, constraint):
        self.boundary.update_handles.append(constraint)

    @property
    def parameters(self):
        return self.boundary.parameters


class FancySystem(engine.OpticalSystem3D):
    def __init__(self, parts):
        """
        A wrapper for a tfrt.engine.OpticalSystem that provides a more friendly interface for the client.

        Parameters
        ----------
        parts : dict of the format <str>: (<str>, <value>)
            A dictionary containing the parts that will go into the optical system.  Keys are the name of each
            part, which can be used to access that part later in the program.  Values are 2-tuples, where the first
            element is a string that labels the type of the part which must be one of {optical, stop, target, source},
            and the second element is the part itself.
        """
        super().__init__()
        self.optical_parts = []
        self.stop_parts = []
        self.target_parts = []
        self.source_parts = []
        self.parts = {}
        self._optical_boundaries = []

        for key, tpl in parts.items():
            category, value = tpl

            if category == "optical":
                self.optical_parts.append(value)
                try:
                    self._optical_boundaries.append(value.boundary)
                except AttributeError:
                    self._optical_boundaries.append(value)
            elif category == "stop":
                self.stop_parts.append(value)
            elif category == "target":
                self.target_parts.append(value)
            elif category == "source":
                self.source_parts.append(value)
            else:
                raise ValueError(f"FancyEngine: category {category} not in {{optical, stop, target, source}}")
            self.parts[key] = value

        self.optical = self._optical_boundaries
        self.stops = self.stop_parts
        self.targets = self.target_parts
        self.sources = self.source_parts
