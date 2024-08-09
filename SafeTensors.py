import safetensors
import tensorflow as tf


def convert_to_safetensors(model_dir, safetensors_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_dir)

    # Extract weights and convert to a dictionary
    weights_dict = {}
    for layer in model.layers:
        for weight in layer.weights:
            weights_dict[weight.name] = weight.numpy()

    # Save to SafeTensors
    safetensors.save_file(weights_dict, safetensors_path)


convert_to_safetensors('my_model', 'model.safetensor')
