import sentencepiece as spm
# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("/Users/sanju/PycharmProjects/Transformers/Tokenizer/tokenizer.model")

# Text to be tokenized
text = "This is a சுவையான உணவு, சுற்றுலா, சுகமான உறக்கம், பதவி என்று ஒவ்வொருவருக்கும் இன்பம் ஒரு வகையில் கிடைக்கும். ஆனால், இந்த இன்பங்கள் எதுவுமே நிலைத்து நிற்பதில்லை. ஆம், பிரியாணியில் இன்பம் கிடைக்கிறது என்பதற்காக மூன்று வேளையும் பிரியாணியே சாப்பிடச்சொன்னால், அதுவே  ."
unknown_token_id = sp.piece_to_id("[UNK]")
print(unknown_token_id)
# Tokenize the text into subword pieces
tokenized_text = sp.encode_as_pieces(text)
print("Tokenized text:", tokenized_text)

# Convert tokenized text into IDs
encoded_ids = sp.encode_as_ids(text)
print("Encoded IDs:", encoded_ids)

encoded_ids_sum = len(encoded_ids)
print("Encoded IDs sum:", encoded_ids_sum)

decoded_text = sp.DecodeIds(encoded_ids)
print(decoded_text)
