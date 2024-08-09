from transformers import AutoTokenizer

# Path to your tokenizer.json file
tokenizer_path = '/Users/sanju/PycharmProjects/Transformers/Tokenizer_vHF'

# Load the tokenizer
#tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print(tokenizer.vocab_size)
# Test the tokenizer with an example sentence
test_sentence = "This is an example சுவையான  sentence."

# Tokenize the sentence
encoded = tokenizer(test_sentence)


# Print the encoded tokens and their corresponding IDs
print("Encoded tokens:", encoded.tokens())
print("Token IDs:", encoded.input_ids)

# Decode the tokens back to the original sentence
decoded_sentence = tokenizer.decode(encoded.input_ids)

# Print the decoded sentence
print("Decoded sentence:", decoded_sentence)
