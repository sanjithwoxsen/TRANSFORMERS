import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, ReLU
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
import numpy as np


# Custom Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :length, :]

    def get_config(self):
        return {
            "d_model": self.d_model,
            "max_len": self.max_len
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom Add & Normalize Layer
class AddNorm(Layer):
    def __init__(self, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        add = x + sublayer_x
        norm = self.layer_norm(add)
        return norm

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


# Custom Feed Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.ff_1 = Dense(d_ff)
        self.ff_2 = Dense(d_model)
        self.relu = ReLU()

    def call(self, x):
        x = self.ff_1(x)
        x = self.relu(x)
        x = self.ff_2(x)
        return x

    def get_config(self):
        return {
            "d_ff": self.ff_1.units,
            "d_model": self.ff_2.units
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_model, d_ff, dropout_rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=h, key_dim=d_model)
        self.ff = FeedForward(d_ff, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()

    def call(self, x, mask, training):
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)

        mha_output = self.mha(x, x, x, attention_mask=mask)
        mha_output = self.dropout1(mha_output, training=training)
        add_norm1_output = self.add_norm1(mha_output, x)
        ff_output = self.ff(add_norm1_output)
        ff_output = self.dropout2(ff_output, training=training)
        encoder_output = self.add_norm2(ff_output, add_norm1_output)
        return encoder_output

    def get_config(self):
        return {
            "h": self.mha.num_heads,
            "d_model": self.mha.key_dim,
            "d_ff": self.ff.ff_1.units,
            "dropout_rate": self.dropout1.rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom Encoder Model
class Encoder(Layer):
    def __init__(self, h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layers = [EncoderLayer(h, d_model, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, mask, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask, training=training)

        return x

    def get_config(self):
        return {
            "h": self.encoder_layers[0].mha.num_heads,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_ff": self.encoder_layers[0].ff.ff_1.units,
            "dropout_rate": self.encoder_layers[0].dropout1.rate,
            "max_len": self.positional_encoding.max_len
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, h, d_model, d_ff, dropout_rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.masked_mha = MultiHeadAttention(num_heads=h, key_dim=d_model)
        self.cross_mha = MultiHeadAttention(num_heads=h, key_dim=d_model)
        self.ff = tf.keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])
        self.dropout = Dropout(dropout_rate)
        self.add_norm1 = LayerNormalization(epsilon=1e-6)
        self.add_norm2 = LayerNormalization(epsilon=1e-6)
        self.add_norm3 = LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, lookahead_mask, training):
        masked_mha = self.masked_mha(query=x, value=x, key=x, attention_mask=lookahead_mask)
        masked_mha = self.dropout(masked_mha, training=training)
        add_norm1 = self.add_norm1(x + masked_mha)

        cross_mha = self.cross_mha(query=add_norm1, value=encoder_output, key=encoder_output)
        cross_mha = self.dropout(cross_mha, training=training)
        add_norm2 = self.add_norm2(add_norm1 + cross_mha)

        ff_output = self.ff(add_norm2)
        ff_output = self.dropout(ff_output, training=training)
        final_output = self.add_norm3(add_norm2 + ff_output)

        return final_output

    def get_config(self):
        return {
            "h": self.masked_mha.num_heads,
            "d_model": self.masked_mha.key_dim,
            "d_ff": self.ff.layers[0].units,
            "dropout_rate": self.dropout.rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom Decoder Model
class Decoder(Layer):
    def __init__(self, h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.decoder_layers = [DecoderLayer(h, d_model, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, encoder_output, lookahead_mask, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, encoder_output, lookahead_mask, training=training)

        return x

    def get_config(self):
        return {
            "h": self.decoder_layers[0].masked_mha.num_heads,
            "vocab_size": self.embedding.input_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_ff": self.decoder_layers[0].ff.layers[0].units,
            "dropout_rate": self.dropout.rate,
            "max_len": self.positional_encoding.max_len
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = Encoder(h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len)
        self.decoder = Decoder(h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len)
        self.final_layer = Dense(vocab_size)
        self.h = h
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.max_len = max_len

    def call(self, inputs, targets, mask=None, lookahead_mask=None, training=False):
        enc_output = self.encoder(inputs, mask, training=training)
        dec_output = self.decoder(targets, enc_output, lookahead_mask=lookahead_mask, training=training)
        final_output = self.final_layer(dec_output)
        return final_output

    def get_config(self):
        config = {
            "h": self.h,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
            "max_len": self.max_len
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

tf.keras.utils.register_keras_serializable()(Transformer)
# Script to build and initialize the model
def build_and_initialize_model():
    h = 8
    vocab_size = 5000
    d_model = 512
    num_layers = 6
    d_ff = 2048
    dropout_rate = 0.1
    max_len = 100

    # Create the Transformer model
    model = Transformer(h, vocab_size, d_model, num_layers, d_ff, dropout_rate, max_len)

    # Define dummy inputs for the model
    example_input = tf.random.uniform(shape=(1, max_len), dtype=tf.int32)
    example_target = tf.random.uniform(shape=(1, max_len), dtype=tf.int32)
    example_mask = tf.ones((1, max_len))
    example_lookahead_mask = tf.ones((1, max_len))

    # Build the model with example inputs
    model(example_input, example_target, mask=example_mask, lookahead_mask=example_lookahead_mask, training=True)

    # Print model summary
    model.summary()

    return model

# Execute the function to build and initialize the model
if __name__ == "__main__":
    build_and_initialize_model()
