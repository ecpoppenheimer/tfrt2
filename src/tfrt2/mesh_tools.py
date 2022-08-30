"""
Various useful mesh processing functions.
"""
import math
import itertools
import time

import numpy as np
import tensorflow as tf
import pyvista as pv


def pack_faces(faces):
    """
    Convert a set of faces (tensor of shape (n, 3)) into the proper format for pymesh:
    each face is prefixed with 3 and the total is flattened.
    """
    faces = np.array(faces, dtype=np.int64)
    return np.reshape(np.pad(faces, ((0, 0), (1, 0)), constant_values=3), (-1,))


def unpack_faces(faces):
    """
    Convert a set of faces from the pymesh format into a 2d array
    (tensor of shape (n, 3)), assuming all faces are triangles.
    """
    return np.reshape(faces, (-1, 4))[:, 1:]


def points_from_faces(vertices, faces):
    first_index, second_index, third_index = tf.unstack(faces, axis=1)
    first_points = tf.gather(vertices, first_index)
    second_points = tf.gather(vertices, second_index)
    third_points = tf.gather(vertices, third_index)

    return first_points, second_points, third_points


def dot(x, y):
    return tf.reduce_sum(x * y, axis=-1)


def np_dot(x, y):
    return np.sum(x * y, axis=-1)


def projection(x, y):
    x, _ = tf.linalg.normalize(x, axis=-1)
    y, _ = tf.linalg.normalize(y, axis=-1)
    return tf.clip_by_value(dot(x, y), -1.0, 1.0)


def np_projection(x, y):
    x /= (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-16)
    y /= (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-16)
    return np.clip(np_dot(x, y), -1.0, 1.0)


def cosine_vum(origin, points, vertices, faces):
    """
    A vertex update map can help speed up training by reducing the number of vertices a face can control to prevent
    adjacent faces from fighting each other.

    It generates a boolean array of the same shape as faces, which is true wherever each face is permitted to pass
    a gradient into each vertex.

    Parameters
    ----------
    origin : float array of shape (3,)
        A single point around which to center the update map.  Often it makes sense to put this at the center of mass
        of the vertices, but different optics could benefit from different locations.
    points : 3-tuple of float array of shape (n, 3)
        The vertices of the mesh to provide this VUM for.  This is a 3-tuple of vertices that have already been
        expanded out of the faces (i.e. all three vertices for each face)
    vertices : float array of shape (m, 3)
        The vertices of the mesh, but not expanded.
    faces : int array of shape (n, 3)
        The faces of the mesh.

    Returns
    -------
    A boolean tensor of shape (m, 3)
    """
    # points is a 3-tuple of individual points, which are just vertices indexed by faces.
    face_centers = (points[0] + points[1] + points[2]) / 3.0
    inner_product = np.stack([np_projection(each - face_centers, face_centers - origin) for each in points], axis=1)
    # could possibly be nans, if any face is on the origin, it will fail to normalize.  In any case this is found, set
    # the element to 1 to allow it to move.
    inner_product = np.where(np.isnan(inner_product), 1.0, inner_product)

    # select movable via selecting two largest ip
    # minimum = np.argmin(inner_product, axis=-1, keepdims=True) # I am having version issues, so cannot use keepdims
    minimum = np.reshape(np.argmin(inner_product, axis=-1), (-1, 1))
    movable = np.tile([[0, 1, 2]], (len(inner_product), 1))
    movable = movable != minimum

    # Check that every face has two movable vertices.  By construction they should, and it any are found without
    # then something has gone wrong.
    movable_count = np.sum(movable, axis=-1)
    assert np.all(movable_count == 2), "mesh_tools.cosine_vum: Got a face without two movable vertices."

    # Check that every vertex has at least one movable, and if there aren't any, set it as movable to at least one face.
    # It would be nice to set it as movable to maybe one face.  But it is super easy to just set it as movable to
    # every face.  This should be a rare occurrence, and I don't want to spend more time developing this, so this
    # is what I am going to do
    vertex_move_count, bins = np.histogram(
        np.where(movable, faces, -1).flatten(),
        bins=np.arange(-1, len(vertices) + 1)
    )
    # -1 is used in the histogram as a sentinel for non-movable vertices.  Have to subtract 1 from the line below
    # to remove this element and find the true indices of the non-movable vertices
    immobile_vertices = np.argwhere(vertex_move_count == 0).flatten() - 1
    not_movable = np.logical_not(movable)
    for v in immobile_vertices:
        highlight = np.logical_and(faces == v, not_movable)
        movable = np.logical_or(movable, highlight)

    return movable


def cosine_acum(origin, vertices):
    size = len(vertices)
    accumulator = np.zeros((size, size), dtype=np.bool)
    for i in range(size):
        accumulator[:, i] = np_projection(origin - vertices[i], vertices - vertices[i]) < -.85
    accumulator = np.logical_or(accumulator, np.eye(size, dtype=np.bool))
    return tf.constant(accumulator, dtype=tf.float64)


def mesh_distance_matrix(mesh):
    """
    Return a matrix that holds the distance between any two points in a pyvista mesh.
    """
    p1 = tf.expand_dims(mesh.points, 0)
    p2 = tf.expand_dims(mesh.points, 1)
    return tf.math.reduce_euclidean_norm(p1 - p2, axis=-1)


def mesh_neighbor_table(mesh):
    """
    Return an array of points adjacent to each point in a pyvista mesh.
    """
    neighbors = [set() for _ in range(mesh.points.shape[0])]
    faces = unpack_faces(mesh.faces)
    for face in faces:
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            neighbors[face[i]] |= {face[j], face[k]}
    return neighbors


# ======================================================================================================================


def circular_mesh(
        radius,
        target_edge_size,
        starting_radius=0,
        theta_start=0,
        theta_end=2 * math.pi,
        join=None
):
    """
    Generate a circular mesh that is as uniform as possible.

    This function is designed for generating zero point meshes for planar lenses.  It works
    well combined with the FromVectorVG vector generator in boundaries.py.  This function
    is limited: it's output is in the x,y plane, and it is centered at zero, so if that isn't
    what you need, you will have to manually rotate the surface after generating it.  Since
    this function returns a pv.PolyData, it will come with built in transformation commands!

    Parameters
    ----------
    radius : float
        The radius of the circle
    target_edge_size : float
        This is approximately how long the edge of each triangle generated by this function
        will be.
    starting_radius : float, optional
        Defaults to zero.  The inner radius of the generated circle.
    theta_start : float, optional
        Defaults to zero.  The angle where the circle starts.
    theta_end : float, optional
        Defaults to 2 PI.  The angle where the circle ends.
    join : bool, optional
        If true, the oppozite sides of the circle will be joined together.  This only makes
        sense for complete circles.  By default the value for this parameter will be inferred
        from theta_start and theta_end, and will default to a joined circle only if the circle
        is complete (i.e. theta_start = 0 and theta_end = 2PI).

    Returns
    -------
    A pv.PolyData in the x,y plane, centered at zero.

    """
    if join is None:
        join = bool(theta_start == 0) and bool(theta_end == 2 * math.pi)

    if starting_radius >= radius:
        raise ValueError("circular_mesh: starting_radius must be < radius.")

    # figure out each radius at which we will place points
    radius_step = target_edge_size * math.sin(math.pi / 3)
    radius_step_count = max(int(1 + (radius - starting_radius) / radius_step), 2)
    radius_steps = np.linspace(starting_radius, radius, radius_step_count)

    # Compute the number of trapezoids each layer will be built from.
    # (including possibly the degenerate trapezoid that is actually a triangle at the
    # center of the wedge).
    trapezoid_count = math.ceil((theta_end - theta_start) / (math.pi / 3))

    # Compute the number of points along the inner edge of each trapezoid.  This will be
    # 1 if we are starting at radius 0.
    if starting_radius != 0:
        starting_arc_length = radius_steps[0] * (theta_end - theta_start) / trapezoid_count
        trapezoid_inner_edge_count = math.ceil(starting_arc_length / target_edge_size) + 1
    else:
        trapezoid_inner_edge_count = 1
    starting_angles = np.linspace(
        theta_start,
        theta_end,
        (trapezoid_inner_edge_count - 1) * trapezoid_count + 1
    )

    # compute the numbers of triangles in each trapezoid, in the first layer
    triangles_per_trapezoid = 2 * trapezoid_inner_edge_count - 1

    # start generating the points
    points = [
        (radius_steps[0] * math.cos(angle), radius_steps[0] * math.sin(angle), 0)
        for angle in starting_angles
    ]
    faces = []
    cumulative_point_count = len(points)
    new_point_count = cumulative_point_count
    last_indices = itertools.cycle(range(cumulative_point_count))
    for radius in radius_steps[1:]:
        # generate the new points
        new_points = []
        new_point_count += trapezoid_count
        for angle in np.linspace(theta_start, theta_end, new_point_count):
            new_points.append((radius * math.cos(angle), radius * math.sin(angle), 0))
        if join:
            # if the edges are joined, remove the last point, because it should be equal
            # to the first point.  Since we are using cycle iterators, this happens
            # automatically!
            new_points.pop()
        points += new_points

        # create iterator for generating the edges.  I am using a cycling iterator to
        # automatically cover the edge case where we need to join the two sides of
        # the wedge when the circle is full.
        new_indices = itertools.cycle(range(
            cumulative_point_count,
            cumulative_point_count + len(new_points)
        ))

        first = next(new_indices)
        second = next(last_indices)
        save_second = second
        for trapezoid in range(trapezoid_count):
            choose_outer = True
            for i in range(triangles_per_trapezoid):
                if choose_outer:
                    third = next(new_indices)
                    faces.append((third, second, first))
                else:
                    third = next(last_indices)
                    faces.append((first, second, third))
                first = second
                save_second = second
                second = third
                choose_outer = not choose_outer
            first = third
            second = save_second

        # update for the next loop pass
        last_indices = itertools.cycle(range(
            cumulative_point_count,
            cumulative_point_count + len(new_points)
        ))
        cumulative_point_count += len(new_points)
        triangles_per_trapezoid += 2

    # process the shape of the faces
    faces = np.array(faces, dtype=np.int64)
    faces = np.pad(faces, ((0, 0), (1, 0)), mode='constant', constant_values=3)
    faces = np.reshape(faces, (-1))

    return pv.PolyData(np.array(points), faces)


def hexagonal_mesh(radius=1.0, step_count=10):
    """
    Generate a totally uniform hexagonal mesh out of equilateral triangles.

    Parameters
    ----------
    radius : float, optional
        The radius of the hexagon (distance between the center and one of the corners).
        Defaults to 1.0
    step_count : int, optional
        Number of layers of triangles to use.
        Defaults to 10

    Returns
    -------
    hexagonal_mesh : pyvista mesh
        The new mesh.

    """
    radius_steps = np.linspace(0, radius, step_count + 1)
    points = [(0.0, 0.0, 0.0)]
    faces = []
    cumulative_point_count = 1
    new_point_count = 1
    last_indices = itertools.cycle([0])

    for radius in radius_steps[1:]:
        # make all of the new points for the entire layer of triangles.

        # create the points along each edge of the hex, but skip the last
        trapezoid_edge_points = [
            np.linspace(
                (radius * np.cos(math.pi / 3 * trapezoid), radius * np.sin(math.pi / 3 * trapezoid), 0.0),
                (radius * np.cos(math.pi / 3 * (trapezoid + 1)), radius * np.sin(math.pi / 3 * (trapezoid + 1)), 0.0),
                new_point_count + 1
            )[:-1, :]
            for trapezoid in range(6)
        ]

        # concatenate the trapezoid edges into a single set of new points.
        new_points = np.concatenate(trapezoid_edge_points, axis=0)
        points = np.concatenate([points, new_points], axis=0)

        # make an iterator for these new points
        new_indices = itertools.cycle(range(
            cumulative_point_count,
            cumulative_point_count + len(new_points)
        ))

        # weave the faces
        first = next(new_indices)
        second = next(last_indices)
        save_second = second
        for trapezoid in range(6):
            choose_outer = True
            for i in range(2 * new_point_count - 1):
                if choose_outer:
                    third = next(new_indices)
                    faces.append((third, second, first))
                else:
                    third = next(last_indices)
                    faces.append((first, second, third))
                first = second
                save_second = second
                second = third
                choose_outer = not choose_outer
            first = third
            second = save_second

        # update things for the next layer pass
        last_indices = itertools.cycle(range(
            cumulative_point_count,
            cumulative_point_count + 6 * new_point_count
        ))
        cumulative_point_count += 6 * new_point_count
        new_point_count += 1

    # process the shape of the faces
    faces = np.array(faces, dtype=np.int64)
    faces = np.pad(faces, ((0, 0), (1, 0)), mode='constant', constant_values=3)
    faces = np.reshape(faces, (-1))

    return pv.PolyData(np.array(points), faces)


# -------------------------------------------------------------------------------------------

def cylindrical_mesh(
        start,
        end,
        radius=1.0,
        theta_res=6,
        z_res=8,
        start_cap=True,
        end_cap=True,
        use_twist=False,
        epsilion=1e-6
):
    """
    Generate a cylindrical mesh suitable for a parametric surface.

    This function is designed to be used to generate a paremtrizable surface that is more
    like a light guide than a lens.  It generates a cylindrical optic between two points.
    The recommended way to use this is to input the minimum allowed radius for the parametric
    optic, constrain the parameters to zero, and initialize them with some non-zero value, if
    you don't want the optic to start at minimum thickness.

    It is recommended to keep both the start and end caps closed, as is default.  This option
    will generate two extra vertices along the axis.  If this surface is used with the
    FromAxisVG vector generator, it should generate zero length vectors for these two
    vertices, and so they should remain immobile, so you don't need to worry about masking them
    out of the gradient calculation.

    Parameters
    ----------
    start : float 3-vector
        The point on the axis where the cylinder will start.
    end : float 3-vector
        The point on the axis where the cylinder will end.
    radius : float
        The radius of the cylinder.  A radius of zero is permissible in case that is how
        you want to parametrize things.
    theta_res : int, optional
        The number of points to place around the diameter of the cylinder.  Defaults to 6.
    z_res : int, optional
        The number of points to place along the axis of the cylinder.  Defaults to 8.
    start_cap : bool, optional
        Defaults to True, in which case triangles are generated to close the start of the
        cylinder.
    end_cap : bool, optional
        Defaults to True, in which case triangles are generated to close the end of the
        cylinder.
    use_twist : bool, optional
        Defaults to True, in which case the triangles will be twisted around the axis each
        layer.  Does not work well for small angular resolution, but could possibly work better
        at high angular resolutions.
    epsilion : float, optional
        A small value to compare to to detect zero length vectors.  Defaults to 1e-6.
        If the distance between end and start is too small, you may need to reduce the size
        of epsilion.

    Returns
    -------
    The cylindrical mesh, a pv.PolyData.

    """
    # reshape everything to (1, 3)
    start = np.reshape(start, (1, 3))
    end = np.reshape(end, (1, 3))
    axis = end - start

    # Need to generate two vectors u, v that are perpendicular to each other and to
    # the axis, and whose length is the radius.
    # Do this by first trying axis X x-hat.  If it is nonzero, good; otherwise try axis X y-
    # hat instead.
    u = np.cross(axis, (1.0, 0.0, 0.0))
    u_norm = np.linalg.norm(u)
    if u_norm < epsilion:
        u = np.cross(axis, (0.0, 1.0, 0.0))
        u_norm = np.linalg.norm(u)
    if u_norm < epsilion:
        raise ValueError(
            "cylindrical_mesh: could not find vectors perpendicular to axis.  Try decreasing "
            "epsilion?"
        )

    u = u * radius / u_norm
    u = np.reshape(u, (1, 3))

    v = np.cross(axis, u)
    v = v * radius / np.linalg.norm(v)
    v = np.reshape(v, (1, 3))

    # parametrize is a function that will convert theta and z parameters into points on the
    # surface of the cylinder
    def parametrize(theta, z):
        return start + z * axis + np.cos(theta) * u + np.sin(theta) * v

    theta, z = np.meshgrid(np.linspace(0, 2 * math.pi, theta_res + 1)[:-1], np.linspace(0, 1, z_res))

    if use_twist:
        # need to twist by half a triangle every layer
        twist = np.reshape(math.pi / theta_res * np.arange(z_res), (-1, 1))
        theta += twist

    # this 3 dimensional array holds all the vertices, except the end caps.  But I want to be
    # careful about the ordering, so I will reorder this manually as I generate the triangles.
    cylinder_points = parametrize(np.expand_dims(theta, 2), np.expand_dims(z, 2))

    points = []
    faces = []

    if start_cap:
        points.append(start[0, :])
        start_offset = 1
    else:
        start_offset = 0

    for theta in range(theta_res):
        points.append(cylinder_points[0, theta])

    if start_cap:
        for theta in range(theta_res):
            faces.append((
                theta + 1,
                0,
                (theta + 1) % theta_res + 1
            ))

    for z in range(1, z_res):
        for theta in range(theta_res):
            points.append(cylinder_points[z, theta])
            faces.append((
                (z - 1) * theta_res + start_offset + (theta + 1) % theta_res,
                z * theta_res + start_offset + theta,
                (z - 1) * theta_res + start_offset + theta
            ))
            faces.append((
                z * theta_res + start_offset + theta,
                (z - 1) * theta_res + start_offset + (theta + 1) % theta_res,
                z * theta_res + start_offset + (theta + 1) % theta_res
            ))

    if end_cap:
        points.append(end[0, :])
        last_vertex = len(points) - 1
        z_offset = (z_res - 1) * theta_res
        if start_cap:
            z_offset += 1
        for theta in range(theta_res):
            faces.append((
                (theta + 1) % theta_res + z_offset,
                last_vertex,
                theta + z_offset
            ))

    return pv.PolyData(np.array(points), pack_faces(faces))


# ======================================================================================================================
# Old and non-working methods from tfrt1.  They worked in tfrt1, so I don't know why I can't get them to work here, but
# I want to re-design these anyway so I am not going to try to get them working at the moment.


def old_mesh_parametrization_tools(vertices, faces, top_parent):
    """
    Raw version of the function, which returns extra information for debugging.

    Determines which of a face's vertices that face is allowed to update, in a way that
    tries to minimize competition between adjacent faces.
    """
    face_sets = [set(face) for face in faces]
    face_count = len(face_sets)
    point_count = vertices.shape[0]
    face_movable_vertices = [set()] * face_count
    faces_to_visit = set(range(face_count))
    active_edge = set([top_parent])
    last_edge = set()
    available_vertices = set(range(point_count))
    vertex_parents = [set()] * point_count
    vertex_ancestors = [set()] * point_count
    missed_vertices = set(range(point_count))

    counter = 0 # to prevent infinite loops
    while faces_to_visit:
        counter += 1
        if counter % 1000 == 0:
            print(f"loop iteration {counter}")
        if counter >= 5:
            break

        next_active_edge = set()
        faces_just_visited = set()
        available_vertices -= active_edge
        # work on face_movable_vertices
        for face in faces_to_visit:
            for vertex in active_edge:
                if vertex in face_sets[face]:
                    movable_vertices = face_sets[face] & available_vertices
                    next_active_edge |= movable_vertices
                    face_movable_vertices[face] = movable_vertices
                    faces_just_visited.add(face)

        # work on vertex ancestors
        # this may miss vertices along the edge after all faces have been depleted.
        for vertex in active_edge:
            missed_vertices.remove(vertex)
            vertex_parents[vertex] = neighbors_from_faces(vertex, face_sets) & last_edge
            vertex_ancestors[vertex] = vertex_parents[vertex].copy()
            for parent in vertex_parents[vertex]:
                vertex_ancestors[vertex] |= vertex_ancestors[parent]

        # step the sets to the next edge
        faces_to_visit -= faces_just_visited
        last_edge = active_edge
        active_edge = next_active_edge

    # take care of any missed vertices
    for vertex in missed_vertices:
        vertex_parents[vertex] = neighbors_from_faces(vertex, face_sets) - missed_vertices
        vertex_ancestors[vertex] = vertex_parents[vertex].copy()
        for parent in vertex_parents[vertex]:
            vertex_ancestors[vertex] |= vertex_ancestors[parent]

    return (
        face_movable_vertices,
        vertex_ancestors,
        vertex_parents,
        missed_vertices
    )


def movable_to_updatable(faces, face_movable_vertices):
    """
    Converts a list of sets that dictates which vertices each face is allowed to move into a boolean array that can
    be used to mask away updates.
    """
    face_updates = []
    face_count = len(face_movable_vertices)

    for face in range(face_count):
        face_updates.append([True if vertex in face_movable_vertices[face] else False for vertex in faces[face]])

    # some faces may have no movable vertices.  We can't really allow this, so instead I am going to allow these to
    # move all of their vertices
    orphaned_count = 0
    for face in range(face_count):
        if not np.any(face_updates[face]):
            orphaned_count += 1
            face_updates[face] = [True] * 3
    if orphaned_count > 0:
        print("Mesh parametrization tools: warning, found orphaned faces in mesh.")

    return np.array(face_updates, dtype=np.bool)


def connections_to_array(connection_list, dtype=np.float64, inverse=True):
    """
    Converts a list of sets of indices that encodes the desired connections into a numpy array that can be used to
    modify the gradient.  The return is a matrix that can be left multiplied onto a vector to implement the desired
    connections upon update.
    """
    size = len(connection_list)
    array = np.array([[1 if j in row else 0 for j in range(size)] for row in connection_list], dtype=dtype)
    array += np.eye(size, dtype=dtype)
    if inverse:
        return array
    else:
        return array.T


def neighbors_from_faces(point, faces):
    """
    Returns a set of all points that are next to point.  Uses faces (formatted as a list of sets) to compute this
    instead of edges.
    """
    out = set()
    for face in faces:
        if point in face:
            out |= face
    out.discard(point)
    return out


def get_closest_point(vertices, target):
    """
    Gets the index of the point in the mesh closest (in cartesian distance) to target.
    """
    distance = np.sum((vertices - target)**2, axis=1)
    return np.argmin(distance)
