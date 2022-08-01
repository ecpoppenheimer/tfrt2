import tensorflow as tf


def normalize(val):
    return tf.linalg.normalize(val, axis=-1)[0]


class FromPointVG:
    """
    Vector generator that projects from a single 3-D point.
    """
    def __init__(self, point):
        self.point = point

    def __call__(self, zero):
        return normalize(zero - self.point)

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, val):
        self._point = tf.cast(val, tf.float64)


class FromVectorVG:
    """
    Vector generator that just copies an input vector.

    The input may be either a single 3-vector, or an (N, 3) tensor which is a list of
    vectors.  If a list of vectors, it is the responsibility of the user to ensure
    that they broadcast to the shape of the points.
    """

    def __init__(self, vector):
        self.vector = vector

    def __call__(self, zero):
        return normalize(tf.broadcast_to(self.vector, tf.shape(zero)))

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, val):
        self._vector = tf.cast(val, tf.float64)


class FromAxisVG:
    """
    Vector generator that projects perpendicular to a line, through points.

    The input will always be two 3-vectors that determine the axis/line that is the
    projection origin.  The first argument is always a point on the line.  The second
    argument is a required keyword argument that is either a second point, or a vector.
    """

    def __init__(self, first, **kwargs):
        self._axis_point = tf.cast(first, tf.float64)

        # interpret the second parameter
        try:
            axis_vector = kwargs["point"] - self.axis_point
        except(KeyError):
            try:
                axis_vector = kwargs["direction"]
            except(KeyError) as e:
                raise ValueError(
                    "FromAxisVG: Must provide a kwarg 'point' or 'direction' to define "
                    "the axis."
                ) from e
        axis_vector = tf.reshape(axis_vector, (1, 3))
        self._axis_vector = normalize(tf.cast(axis_vector, tf.float64))

    def __call__(self, zero):
        axis_vector = tf.broadcast_to(self.axis_vector, tf.shape(zero))
        d = tf.reduce_sum((zero - self.axis_point) * axis_vector, axis=-1)
        closest_point = self.axis_point + axis_vector * tf.reshape(d, (-1, 1))

        return normalize(zero - closest_point)

    @property
    def axis_point(self):
        return self._axis_point

    @axis_point.setter
    def axis_point(self, val):
        self._axis_point = tf.cast(val, tf.float64)

    @property
    def axis_vector(self):
        return self._axis_vector

    @axis_vector.setter
    def axis_vector(self, val):
        self._axis_vector = normalize(tf.cast(val, tf.float64))
