import tensorflow as tf

# Define a simple MultiHeadAttention layer
num_heads = 8
d_model = 128

multi_head_attention = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads, key_dim=d_model // num_heads
)

# Example input tensors
query = tf.random.normal((1, 5, d_model))  # (batch_size, seq_len, d_model)
key = tf.random.normal((1, 5, d_model))
value = tf.random.normal((1, 5, d_model))

# Create a look-ahead mask
def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = (mask - 1) * 1e9
    return mask

look_ahead_mask = create_look_ahead_mask(5)  # For a sequence length of 5

# Apply the mask to the attention layer
attention_output = multi_head_attention(
    query, key, value,
    attention_mask=look_ahead_mask
)

print(attention_output.shape)
print(attention_output[0, :, :])

print(po)