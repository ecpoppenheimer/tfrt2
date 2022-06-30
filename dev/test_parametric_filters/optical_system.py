import numpy as np
import tensorflow as tf
import pyvista as pv
import pathlib

import tfrt2.src.fancy_optics as fancy_optics
import tfrt2.src.settings as settings
import tfrt.materials as materials
import tfrt.sources as sources
import tfrt.distributions as distributions
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
            lens=settings.Settings()
        )
        self.settings.lens.establish_defaults(
            visible=True,
            color="green",
            show_edges=True,
            mesh_output_path=str(pathlib.Path(path_to_self) / "test_output.stl"),
            mesh_input_path=str(pathlib.Path(path_to_self) / "test_input.stl"),
        )

        PARALLEL_AXIS = 1
        PERPENDICULAR_AXIS = 0

        def filter_fixed(vertices):
            s = vertices[:, PARALLEL_AXIS]
            min, max = np.amin(s), np.amax(s)
            return np.logical_or(s > max - .001, s < min + .001)

        def filter_drivers(vertices):
            s = vertices[:, PERPENDICULAR_AXIS]
            min = np.amin(s)
            return s < min + .001

        def attach_to_driver(v, vertices):
            return np.abs(vertices[:, PARALLEL_AXIS] - v[PARALLEL_AXIS]) < .001

        lens = optics.ParametricTriangleOptic(
            "exit_window",
            vg.FromVectorVG((0.0, 0.0, 1.0)),
            self.client,
            path_to_self,
            mesh=pv.Plane(i_resolution=9, j_resolution=9).triangulate(),
            settings=self.settings.lens,
            mat_in=1,
            mat_out=0,
            #filter_fixed=filter_fixed,
            #filter_drivers=filter_drivers,
            #attach_to_driver=attach_to_driver
        )
        lens.update()

        # package the parts
        parts = {
            "lens": ("optical", lens),
        }
        self.build_from_parts(parts)

        #lens.controller_widgets = [cpw.OpticController(lens)]
        #exit_window.controller_widgets = [cpw.OpticController(exit_window)]

        self.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
        self.update()
