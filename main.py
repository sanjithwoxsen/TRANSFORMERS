import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Input
from tensorflow.keras.models import Model
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_parquet('train-00000-of-00001.parquet')
tamil_data = df['tamil'].tolist()
english_data = df['english'].tolist()


# Function to train SentencePiece tokenizer
def create_sentencepiece_tokenizer(tamil_data, english_data, vocab_size=8000):
    with open('combined_data.txt', 'w', encoding='utf-8') as f:
        for sentence in tamil_data + english_data:
            f.write(f'{sentence}\n')

    SentencePieceTrainer.Train(
        f"--input=combined_data.txt --model_prefix=tamil_english_spm --character_coverage=1.0 --vocab_size={vocab_size}"
    )
    spm = SentencePieceProcessor()
    spm.load("tamil_english_spm.model")
    return spm


# Preprocess data function
def preprocess_data(tokenizer, sentences):
    tokenized_sentences = [tokenizer.encode_as_ids(sentence) for sentence in sentences]
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentences, padding='post')
    return padded_sequences


# Custom layer to create look-ahead mask
class LookAheadMask(tf.keras.layers.Layer):
    def call(self, x):
        size = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


# Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Transformer Encoder
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return x


# Transformer Decoder Layer
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


# Transformer Decoder
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask,
                                   padding_mask=padding_mask)
        return x


# Full Transformer Model
def create_transformer_model(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
    inputs = Input(shape=(None,))
    dec_inputs = Input(shape=(None,))

    look_ahead_mask = LookAheadMask()(dec_inputs)
    padding_mask = None  # Assuming padding mask is not required for encoder output in this context

    encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
    decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

    enc_output = encoder(inputs, training=True)
    dec_output = decoder(dec_inputs, enc_output, training=True, look_ahead_mask=look_ahead_mask,
                         padding_mask=padding_mask)

    final_output = Dense(target_vocab_size)(dec_output)  # Remove softmax activation here
    final_output = tf.keras.layers.Softmax(axis=0)(final_output)  # Apply softmax to the last axis
    model = Model(inputs=[inputs, dec_inputs], outputs=final_output)
    return model


# Train SentencePiece tokenizer
tokenizer = create_sentencepiece_tokenizer(tamil_data, english_data)
tamil_vocab_size = tokenizer.get_piece_size()  # Get vocabulary size from tokenizer

# Preprocess data
tamil_sequences = preprocess_data(tokenizer, tamil_data)
english_sequences = preprocess_data(tokenizer, english_data)

# Define and compile the model
model = create_transformer_model(num_layers=1, d_model=64, num_heads=8, dff=256,
                                 input_vocab_size=tamil_vocab_size, target_vocab_size=tamil_vocab_size)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([tamil_sequences, english_sequences[:, :-1]], english_sequences[:, 1:], epochs=3)


# Function to translate a sentence
def translate(sentence):
    # Tokenize and pad the input sentence
    tokenized_sentence = tokenizer.encode_as_ids(sentence)
    input_ids = tf.keras.preprocessing.sequence.pad_sequences([tokenized_sentence], padding='post')

    # Predict the output
    prediction = model.predict([input_ids, np.zeros((1, 1))])

    # Convert predictions to token IDs
    predicted_token_ids = np.argmax(prediction, axis=-1)

    # Decode the token IDs to get the translated sentence
    translated_sentence = tokenizer.decode(predicted_token_ids[0])

    return translated_sentence


# Example usage
translated_text = translate("நாம் இப்போது நோயற்ற 4 மாத வயது மேலான எலி வகைகளை வைத்திருக்கின்றோம், அவை முன்பு நிலைமையானவை ஆகும், என்று அவர் சேர்த்துக்கொண்டார்")
print(translated_text)