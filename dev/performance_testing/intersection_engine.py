import time

import tensorflow as tf

import tfrt2.trace_engine as engine


def test_raw(count, dtype):
    count = int(count)
    rays = tf.random.uniform((count, 6), -1.0, 1.0, dtype=dtype)
    boundaries = tf.random.uniform((count, 9), -1.0, 1.0, dtype=dtype)
    epsilon = tf.constant(1e-10, dtype=dtype)

    t = time.time()
    engine.raw_line_triangle_intersection(rays, boundaries, epsilon)
    print(f"running raw_line_triangle_intersection took {time.time() - t} s for {count} rays and {dtype}.")


for tp in tf.float32, tf.float64:
    for sz in 1e4, 1e6, 1e7:
        test_raw(sz, tp)
