import tensorflow as tf
from transformers import AutoTokenizer
from Transformers_Architecture import Transformer
import pandas as pd
def create_lookahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


tf.keras.utils.register_keras_serializable()(Transformer)


def load_model(json_path, weights_path):
    # Load the model architecture from JSON
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()

    # Use model_from_json to load the architecture
    model = tf.keras.models.model_from_json(model_json, custom_objects={'Transformer': Transformer})

    # Load weights into the new model
    model.load_weights(weights_path)

    return model


# Example usage
model = load_model('transformer_model.json', 'transformer.weights.h5')
# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained('/Users/sanju/PycharmProjects/Transformers/Tokenizer_vHF')
a = model.summary()
print(a)
# Special Tokens
Start_token = "<s>"
End_token = "</s>"

# Sample English input
data = pd.read_parquet("train-00000-of-00001.parquet")
data = data["english"].tolist()

english_text = "Hi"
print(english_text)
# Tokenize the input
inputs = tokenizer(english_text, return_tensors='tf', padding=True, truncation=True)

input_ids = inputs['input_ids']
input_attention_mask = inputs['attention_mask']


# Define a function to perform inference
def generate_translation(model, input_ids, input_attention_mask, max_len=100):
    # Create an empty target sequence with the start token
    start_token_id = tokenizer.convert_tokens_to_ids(Start_token)
    target_seq = tf.expand_dims([start_token_id], 0)

    for i in range(max_len):
        lookahead_mask = create_lookahead_mask(target_seq.shape[1])

        # Predict the next token
        predictions = model(inputs=input_ids, targets=target_seq, mask=input_attention_mask,
                            lookahead_mask=lookahead_mask, training=False)
        next_token_logits = predictions[:, -1, :]

        # Get the ID of the most probable next token
        next_token_id = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)

        # Append the predicted token to the target sequence
        target_seq = tf.concat([target_seq, tf.expand_dims(next_token_id, 0)], axis=-1)

        # Stop if the end token is generated
        if next_token_id == tokenizer.convert_tokens_to_ids(End_token):
            break

    return target_seq


# Generate translation
translated_ids = generate_translation(model, input_ids, input_attention_mask)

# Decode the token IDs to text
translated_text = tokenizer.decode(translated_ids.numpy()[0], skip_special_tokens=False)
print("Translated text:", translated_text)
a = model.summary()
print(a)