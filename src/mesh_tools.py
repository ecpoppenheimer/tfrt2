"""
Various useful mesh processing functions.
"""
import numpy as np
import tensorflow as tf


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

    It generates an boolearn array of the same shape as faces, which is true wherever each face is permitted to pass
    a gradient into each vertex.

    Parameters
    ----------
    origin : float tensor of shape (3,)
        A single point around which to center the update map.  Often it makes sense to put this at the center of mass
        of the vertices, but different optics could benefit from different locations.
    points : 3-tuple of float tensor of shape (n, 3)
        The vertices of the mesh to provide this VUM for.  This is a 3-tuple of vertices that have already been
        expanded out of the faces (i.e. all three vertices for each face)
    vertices : float tensor of shape (m, 3)
        The vertices of the mesh, but not expanded.
    faces : int tensor of shape (n, 3)
        The faces of the mesh.

    Returns
    -------
    A boolean tensor of shape (m, 3)
    """
    # points is a 3-tuple of individual points, which are just vertices indexed by faces.
    face_centers = (points[0] + points[1] + points[2]) / 3.0
    inner_product = tf.stack([projection(each - face_centers, face_centers - origin) for each in points], axis=1)
    # could possibly be nans, if any face is on the origin, it will fail to normalize.  In any case this is found, set
    # the element to 1 to allow it to move.
    inner_product = tf.where(tf.math.is_nan(inner_product), 1.0, inner_product).numpy()

    # select movable via selecting two largest ip
    minimum = np.argmin(inner_product, axis=-1, keepdims=True)
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

    return tf.constant(movable, dtype=tf.bool)


def cosine_acum(origin, vertices):
    size = len(vertices)
    accumulator = np.zeros((size, size), dtype=np.bool)
    for i in range(size):
        accumulator[:, i] = np_projection(origin - vertices[i], vertices - vertices[i]) < -.85
    accumulator = np.logical_or(accumulator, np.eye(size, dtype=np.bool))
    return tf.constant(accumulator, dtype=tf.float64)


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
    Gets the index of the point in the mesh closest (in cartesian distance) to traget.
    """
    distance = np.sum((vertices - target)**2, axis=1)
    return np.argmin(distance)
