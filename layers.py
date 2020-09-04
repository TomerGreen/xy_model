import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Lambda


# class SliceRepetitions(Layer):
#
#     def __init__(self, final_size, num_reps):
#         super(SliceRepetitions, self).__init__()
#         self.num_reps = num_reps
#         self.final_size = final_size
#
#     def __call__(self, inputs, *args, **kwargs):
#         input_shape = tf.shape(inputs)
#         batch_size, inputs_len = input_shape[0], input_shape[1]
#         begin = inputs_len * tf.cast(tf.floor(self.num_reps / 2), tf.int32)
#         size = tf.cast(inputs_len / self.num_reps, tf.int32)
#         print(inputs)
#         output = tf.slice(inputs, [0, begin], [batch_size, size])
#         print(output)
#         # Workaround for compute_output_shape not being called
#         output = tf.reshape(output, self.compute_output_shape(input_shape))
#         print(output)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.final_size


def SliceRepetitions(final_size, num_reps):

    def slice_repetitions(inputs):
        input_shape = tf.shape(inputs)
        output = tf.slice(inputs, [0, final_size * int(num_reps / 2)], [input_shape[0], final_size])
        # Workaround for compute_output_shape not being called
        output = tf.reshape(output, compute_output_shape(input_shape))
        return output

    def compute_output_shape(input_shape):
        return input_shape[0], final_size

    return Lambda(slice_repetitions, name='slice_repetitions')

