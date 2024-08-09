import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, ReLU
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from transformers import AutoTokenizer
import numpy as np


# Custom Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len or 5000

        # Create positional encoding matrix
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        print(self.pos_encoding.shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :length, :]


# Custom Add & Normalize Layer
class AddNorm(Layer):
    def __init__(self, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        add = x + sublayer_x
        norm = self.layer_norm(add)
        return norm


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
        # Ensure mask is correctly shaped
        mask = tf.expand_dims(mask, axis=1)  # Expand dimension for num_heads

        mask = tf.expand_dims(mask, axis=1)  # Expand dimension for batch_size

        mha_output = self.mha(x, x, x, attention_mask=mask)
        mha_output = self.dropout1(mha_output, training=training)
        add_norm1_output = self.add_norm1(mha_output, x)
        ff_output = self.ff(add_norm1_output)
        ff_output = self.dropout2(ff_output, training=training)
        encoder_output = self.add_norm2(ff_output, add_norm1_output)
        return encoder_output


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

    def call(self, x, mask, training):  # Ensure training is set as a keyword argument
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask, training=training)

        return x


# Example usage with tokenizer
def main():
    # Example dataset
    texts = [
        "This is the first text.",
        "Here's the second one.",
        "And here is another example.",
        "This is an example சுவையான உணவு sentence.",
        "And here is another example.",
        "This is an example சுவையான உணவு sentence."
    ]

    # Initialize and fit tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/Users/sanju/PycharmProjects/Transformers/Tokenizer_vHF')
    sequences = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

    input_ids = sequences["input_ids"]
    attention_mask = sequences["attention_mask"]
    print(input_ids)
    print(attention_mask)
    #print(attention_mask)
    print(tf.shape(input_ids)[0])

    # Model parameters
    h = 16
    vocab_size = tokenizer.vocab_size
    d_model = 1024
    num_layers = 12
    d_ff = 4096 # dff = 2 * d_model (or) 4 * d_model
    dropout_rate = 0.2

    # Instantiate the custom Encoder model
    encoder = Encoder(h, vocab_size, d_model, num_layers, d_ff, dropout_rate,6000)

    # Test with example input
    output = encoder(input_ids, mask=attention_mask, training=False)  # Pass training as keyword argument
    print(output)

    print(encoder.count_params())
    # Print output shape
    print("Output Shape:", output.shape)


if __name__ == "__main__":
    main()
