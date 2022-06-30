"""
Utilities for drawing optical elements with matplotlib.

This module assists in visualizing the raysets and optical systems built with tfrt.
These classes act as a nice interface that connects ndarrays formatted like tfrt
optical elements to a set of matplotlib axes so that they can be displayed in a
matplotlib figure.

Please note that the optical system data objects fed to these classes do not have to 
be numpy ndarrays, but it is highly recommended that they be.  They must at least 
have a shape attribute and the proper shape requirements to represent that kind of 
object (see tfrt.raytrace for details on the required shapes).  TensorFlow tensors 
are not acceptable inputs to these classes, but the arrays returned by session.run
calls are.

This module defines some helpful constants which are the values of the wavelength of
visible light of various colors, in um.  These values give nice results when using
the default colormap to display rays, but they are not used anywhere in this module.
They exist only as convenient reference points for users of this module, and if you
need different wavelenghts for your application, you can use whatever values you want.

Most changes made to the drawing classes defined in this module will require the 
mpl canvas to be redrawn in order to see the change.  A convenience function,
redraw_current_figure() is provided to do this.  Multiple changes to the drawing
classes can be chained with only a single canvas redraw, in general you should use
as few canvas redraws as possible.

All drawing classes define a draw method, which updates the mpl artists controlled
by the drawing class.  Changes to some class attributes of the drawing classes 
require that the draw method be called again to visualize the change, others do not.
Each attribute is explicitly labeled whether it requires a redraw or not.  Calling a 
class draw method does not redraw the mpl canvas.

"""

import numpy as np
import matplotlib as mpl
import pyvista as pv
import tensorflow as tf

import tfrt2.src.wavelength as wavelength
from tfrt.boundaries import TriangleBoundaryBase


class RayDrawer3D:
    """
    Class for drawing a rayset.

    This class makes it easy to draw a set of rays to a pyvista.Plotter.  By 
    default this class will use the spectrumRGB colormap to color the rays by 
    wavelength, but a different colormap can be chosen if desired.

    Parameters
    ----------
    plot : pyvista.Plotter
        A handle to the pyvista plotter into which the rays will be drawn.
    rays : np.ndarray
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", 
        "wavelength".
    min_wavelength : float, optional
        The minimum wavelength, used only to normalize the colormap.
    max_wavelength : float, optional
        The maximum wavelength, used only to normalize the colormap.
    colormap : matplotlib.colors.Colormap, optional
        The colormap to use for coloring the rays.  Defaults to the spectrumRGB map.
    
    Public attributes
    -----------------
    rays : dict
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", 
        "wavelength".  Requires class redraw.
    plot : matplotlib.axes.Axes
        A handle to the pyvista plotter into which the rays will be drawn.
    colormap : matplotlib.colors.Colormap
        The colormap to use for coloring the rays.
        
    Public members
    --------------
    set_wavelength_limits(min, max)
        Change the minimum and maximum wavelengths for colormap normalization.
        Requires class redraw.
    """

    def __init__(
        self,
        plot,
        rays=None,
        min_wavelength=wavelength.VISIBLE_MIN,
        max_wavelength=wavelength.VISIBLE_MAX,
        colormap=mpl.colors.ListedColormap(wavelength.rgb()),
    ):

        self.plot = plot
        self._rays = rays
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self._colormap = colormap
        self._ray_signature = set([
            "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "wavelength"
        ])

        self._mesh = None
        self._actor = None

    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays):
        if bool(rays):
            try:
                if self._ray_signature <= rays.keys():
                    self._rays = rays
                else:
                    raise ValueError(
                        f"RayDrawer: Rays does not have the proper"
                        " signature."
                    )
                    
            except AttributeError as e:
                raise ValueError(
                    f"RayDrawer: Rays doesn't have a signature."
            ) from e
        else:
            self._rays = {}

    def draw(self):
        """Redraw the pyvista actor controlled by this class."""
        if bool(self.rays) and tf.greater(tf.shape(self.rays['x_start'])[0], 0):
            start_points = tf.stack(
                [self.rays[field] for field in ("x_start", "y_start", "z_start")],
                axis=1
            )
            end_points = tf.stack(
                [self.rays[field] for field in ("x_end", "y_end", "z_end")], 
                axis=1
            )
            all_points = tf.concat([start_points, end_points], 0)
            line_count = tf.shape(self.rays["x_start"])[0]
            cell_range = tf.range(2 * line_count)
            cells = tf.stack([
                2 * tf.ones((line_count,), dtype=tf.int32),
                cell_range[:line_count],
                cell_range[line_count:]
            ], axis=1)

            if self._mesh is None:
                self._mesh = pv.PolyData()
            self._mesh.points = all_points.numpy()
            self._mesh.lines = cells.numpy()
            self._mesh["wavelength"] = self.rays["wavelength"]
            
            self._actor = self.plot.add_mesh(
                self._mesh,
                cmap=self._colormap,
                clim=(self.min_wavelength, self.max_wavelength),
                reset_camera=False
            )
        else: # nothing to draw
            if self._actor is not None:
                self.plot.remove_actor(self._actor)
                self._mesh = None

    def delete(self):
        self.plot.remove_actor(self._norm_actor)
        self.plot.remove_actor(self._parameter_actor)
        self.plot.remove_actor(self._actor)


class TriangleDrawer:
    """
    Class for drawing pyvista meshes to a pyvista plot.
    
    Contains utilities for drawing norm arrows, and possibly parameter vectors, if the
    boundary is parametric.
    
    Unlike the other drawers, this class will draw the pyvista.PolyData mesh held by the 
    boundary given to it - it will not interpret data from the boundary's fields.  Which 
    means you may need to call update_mesh_from_vertices() before drawing the boundary if 
    the boundary is parametric, in order to get the properly updated version of the 
    boundary.  This also means it is impossible to draw an amalgamated boundary with a
    single drawer - you will have to use multiple drawers to draw multiple boundaries, or
    multiple layers of a multi-boundary.
    
    Unlike the other drawers, you don't necessairly need to call draw() every time you need
    to update the display, since pyplot automatically redraws when one of its plotted meshes
    change.  But if you are using a parametric boundary, you will need to use the boundary's 
    update_mesh_from_vertices() every time the parameter changes, to update the mesh.
    """
    def __init__(
        self,
        plot,
        component,
        norm_arrow_visibility=False,
        norm_arrow_length=0.1,
        parameter_arrow_visibility=False,
        parameter_arrow_length=0.1,
        visible=True,
        color="gray",
        show_edges=False
    ):
        self.plot = plot
        self.component = component
        self.visible = visible
        self.norm_arrow_visibility = norm_arrow_visibility
        self.norm_arrow_length = norm_arrow_length
        self.parameter_arrow_visibility = parameter_arrow_visibility
        self.parameter_arrow_length = parameter_arrow_length
        self._actor = None
        self._norm_actor = None
        self._parameter_actor = None
        self.color = color
        self.show_edges = show_edges
                
    def _draw_norm_arrows(self):
        self.plot.remove_actor(self._norm_actor)
        if self.norm_arrow_visibility:
            faces = self.component.faces
            vertices = self.component.vertices
            first_index, pivot_index, second_index = tf.unstack(faces, axis=1)
            pivot_points = tf.gather(vertices, pivot_index)
            first_points = tf.gather(vertices, first_index)
            second_points = tf.gather(vertices, second_index)
            points = (pivot_points + first_points + second_points)/3

            self._norm_actor = self.plot.add_arrows(
                points.numpy(),
                self.component["norm"].numpy(),
                mag=self.norm_arrow_length,
                color=self.color,
                reset_camera=False
            )
                
    def _draw_parameter_arrows(self):
        self.plot.remove_actor(self._parameter_actor)
        if self.parameter_arrow_visibility:
            if hasattr(self.component, "vectors"):
                self._parameter_actor = self.plot.add_arrows(
                    self.component.vertices.numpy(),
                    self.component.vectors.numpy(),
                    mag=self.parameter_arrow_length,
                    color=self.color,
                    reset_camera=False
                )
            
    def draw(self):
        self.plot.remove_actor(self._norm_actor)
        self.plot.remove_actor(self._parameter_actor)
        self.plot.remove_actor(self._actor)
        if self.visible:
            # draw the mesh itself
            self._actor = self.plot.add_mesh(
                self.component.as_mesh(),
                color=self.color,
                show_edges=self.show_edges,
                reset_camera=False
            )
                
            # draw the norm arrows
            self._draw_norm_arrows()
            self._draw_parameter_arrows()

    def delete(self):
        self.plot.remove_actor(self._norm_actor)
        self.plot.remove_actor(self._parameter_actor)
        self.plot.remove_actor(self._actor)


class GoalDrawer3D:
    """
    Class for visualizing the training goal in 3D, using pyvista.
    
    Accepts two sets of points, the resultant output from ray tracing and the optimization 
    goal points, and draws arrows from the output to the goal.  Display can be turned on and 
    off.
    """
    
    def __init__(self, plot, visible=True):
        """
        plot : pyvista plot
            The plot object to plot into
        visible : bool, optional
            The starting state of the visibility of the goal visualization
        """
        self.plot = plot
        self.output = None
        self.goal = None
        self._arrows = None
        self._visible = visible
        
    @property
    def output(self):
        return self._output
        
    @output.setter
    def output(self, val):
        if val is not None:
            val = np.array(val)
            if len(val.shape) != 2 or val.shape[1] != 3:
                raise ValueError("GoalDrawer3D: output must have shape (None, 3).")
        self._output = val
        
    @property
    def goal(self):
        return self._goal
        
    @goal.setter
    def goal(self, val):
        if val is not None:
            val = np.array(val)
            if len(val.shape) != 2 or val.shape[1] != 3:
                raise ValueError("GoalDrawer3D: goal must have shape (None, 3).")
        self._goal = val
        
    @property
    def visible(self):
        return self._visible
        
    @visible.setter
    def visible(self, val):
        if type(val) is bool:
            self._visible = val
            self.draw()
        else:
            raise ValueError("GoalDrawer3D: visible must be a bool.")
            
    def draw(self):
        self.plot.remove_actor(self._arrows)
        if self._output is not None and self._goal is not None:
            if self._visible:
                outputs = np.array(self._output)
                goals = np.array(self._goal)
                
                points = pv.PolyData(outputs)
                points.vectors = goals - outputs
                self._arrows = self.plot.add_mesh(points.arrows, reset_camera=False)
                    
                return
        self._arrows = None
