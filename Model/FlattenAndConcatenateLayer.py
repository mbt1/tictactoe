import tensorflow as tf

class FlattenAndConcatenateLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FlattenAndConcatenateLayer, self).__init__()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        flattened_inputs = [self.flatten(input_tensor) for input_tensor in inputs]
        concatenated_output = tf.keras.layers.Concatenate(axis=-1)(flattened_inputs)
        return concatenated_output
