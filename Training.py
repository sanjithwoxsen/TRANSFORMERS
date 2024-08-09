import tensorflow as tf
from transformers import AutoTokenizer
from Transformers_Architecture import Transformer
import pandas as pd
from tqdm import tqdm


def create_lookahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained('/Users/sanju/PycharmProjects/Transformers/Tokenizer_vHF')

#Special Tokens
Start_token = "<s>"
End_token = "</s>"

# Prepare the data
data = pd.read_parquet("train-00000-of-00001.parquet")
data["tamil"] = Start_token + data["tamil"] + End_token
english_texts = data["english"].tolist()[::800]
tamil_texts = data["tamil"].tolist()[::800]

inputs = tokenizer(english_texts, return_tensors='tf', padding=True, truncation=True)
targets = tokenizer(tamil_texts, return_tensors='tf', padding=True, truncation=True)

input_ids = inputs['input_ids']
target_ids = targets['input_ids']

input_attention_mask = inputs['attention_mask']
target_attention_mask = targets['attention_mask']

# Create Lookahead Mask
lookahead_mask = create_lookahead_mask(target_ids.shape[1])
print(lookahead_mask)

# Instantiate the Transformer model
num_attention_heads = 8
vocab_size = tokenizer.vocab_size
d_model = 64
num_layers = 10
d_ff = 256
dropout_rate = 0.1
max_len = 128

model = Transformer(
    h=num_attention_heads,
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    d_ff=d_ff,
    dropout_rate=dropout_rate,
    max_len=max_len
)

# Define loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
epochs = 2

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Create a progress bar
    with tqdm(total=len(input_ids), desc=f"Training Epoch {epoch + 1}") as pbar:
        for batch in range(len(input_ids)):
            with tf.GradientTape() as tape:
                logits = model(
                    inputs=input_ids[batch:batch + 1],
                    targets=target_ids[batch:batch + 1],
                    mask=input_attention_mask[batch:batch + 1],
                    lookahead_mask=lookahead_mask,
                    training=True
                )
                loss = loss_fn(target_ids[batch:batch + 1], logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update progress bar
            pbar.set_postfix({"Loss": loss.numpy()})
            pbar.update(1)

    print(f"Epoch {epoch + 1} completed. Loss: {loss.numpy()}")

a = model.summary()
print(a)
def save_model(model, architecture_path, weights_path):
    # Save the architecture to JSON
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())

    # Save weights to H5 file
    model.save_weights(weights_path)

# Example usage
if __name__ == "__main__":
    save_model(model, 'transformer_model1.json', 'transformer1.weights.h5')
