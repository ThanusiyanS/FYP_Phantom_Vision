# yamnet_layer.py

import tensorflow as tf
import tensorflow_hub as hub

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

@tf.keras.utils.register_keras_serializable()
class YamnetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yamnet = yamnet_model

    def call(self, inputs):
        def extract_fn(waveform):
            _, embedding, _ = self.yamnet(waveform)
            return tf.reduce_mean(embedding, axis=0)

        return tf.map_fn(
            extract_fn,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(1024,), dtype=tf.float32)
        )
