from tokenizers import SentencePieceBPETokenizer
import json
import transformers
import os
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


class Tokenizer():
    def __init__(self, text_data):
        super(Tokenizer, self).__init__()
        self.text = text_data

    def __sentencepiece_tokenizer(self, vocab_size, output_dir):
        combined_data_file = os.path.join(output_dir, 'combined_data.txt')
        with open(combined_data_file, 'w', encoding='utf-8') as f:
            for sentence in self.text:
                f.write(f'{sentence}\n')

        model_prefix = os.path.join(output_dir, 'tamil_english_spm')
        SentencePieceTrainer.Train(
            f"--input={combined_data_file} --model_prefix={model_prefix} --character_coverage=1.0 --vocab_size={vocab_size} --unk_piece=[UNK]"
        )

        spm = SentencePieceProcessor()
        spm.load(f"{model_prefix}.model")

        with open(f"{model_prefix}.vocab", 'r', encoding='utf-8') as vocab_file:
            vocab = {line.split('\t')[0]: i for i, line in enumerate(vocab_file)}

        with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as vocab_json:
            json.dump(vocab, vocab_json, ensure_ascii=False)

        os.rename(f"{model_prefix}.model", os.path.join(output_dir, 'tokenizer.model'))

        return spm

    def spm_tokenizer(self , vocab_size):

        output_dir = 'Tokenizer_vM'
        os.makedirs(output_dir, exist_ok=True)
        spm = self.__sentencepiece_tokenizer(vocab_size=vocab_size, output_dir=output_dir)

        tokenizer_config = {
            "vocab_size": vocab_size,
            "model_type": "SentencePiece",
            "special_tokens_map": {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "mask_token": "[MASK]"
            }
        }

        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(tokenizer_config, config_file, ensure_ascii=False)

        special_tokens_map = {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]"
        }

        with open(os.path.join(output_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as special_tokens_file:
            json.dump(special_tokens_map, special_tokens_file, ensure_ascii=False)

        added_tokens = {
            # Example: "new_token": 10000
        }

        with open(os.path.join(output_dir, 'added_tokens.json'), 'w', encoding='utf-8') as added_tokens_file:
            json.dump(added_tokens, added_tokens_file, ensure_ascii=False)
        print(f"Default Sentence Piece Tokenization Complete.Stores in {output_dir}")


    def Transformers_Tokenizer(self,vocab_size):
        print("Training SentencePiece...With \"Transformers\" Supported Fast Tokenization")
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
        tk_tokenizer = SentencePieceBPETokenizer()
        tk_tokenizer.train_from_iterator(
            self.text,
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=True,
            special_tokens=special_tokens
        )

        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer, special_tokens=special_tokens)
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
        tokenizer.cls_token = "<cls>"
        tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
        tokenizer.sep_token = "<sep>"
        tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
        # and save for later!
        output_dir_hf = 'Tokenizer_vHF'
        os.makedirs(output_dir_hf, exist_ok=True)
        tokenizer.save_pretrained(output_dir_hf)
        print(f"Transformers (Hugging Face) supported Fast_Tokenization Sentence Piece Tokenization Complete. Stored in {output_dir_hf}")
