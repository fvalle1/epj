import tensorflow as tf

@tf.function
def kullbach_liebler(theta_k, theta_l):
    # dey-visualizing paper
    return tf.subtract(tf.add(tf.math.multiply(theta_k, tf.math.log(tf.math.divide(theta_k, theta_l))), theta_l),
                       theta_k)

@tf.function
def distinctivness(theta_k):
    return tf.reduce_min(tf.sort(theta_k)[:, 1:], axis=1)

@tf.function
def get_distinctivness(data):
    KL_tensor = tf.map_fn(fn=lambda k: tf.map_fn(fn=lambda l: kullbach_liebler(k, l), elems=data), elems=data, parallel_iterations=3)
    KL_tensor_min = tf.map_fn(distinctivness, tf.transpose(KL_tensor, perm=[2, 0, 1]), parallel_iterations=3)
    return KL_tensor_min