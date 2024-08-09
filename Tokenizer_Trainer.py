import pandas as pd
from Tokenizer_Arch import Tokenizer
def separate_lists(filename):
    all_lists = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            all_lists.append(words)
    return all_lists
df = pd.read_parquet('train-00000-of-00001.parquet')
df2 = pd.read_parquet('train-00000-of-00005.parquet')
df3 = pd.read_parquet('train-00001-of-00005.parquet')
df4 = pd.read_parquet('train-00002-of-00005.parquet')
df5 = pd.read_parquet('train-00003-of-00005.parquet')
df6 = pd.read_parquet('train-00004-of-00005.parquet')
df7 = separate_lists("words.txt")

tamil_data1 = df2['text'].tolist()
tamil_data2 = df['tamil'].tolist()
tamil_data3 = df3['text'].tolist()
tamil_data4 = df4['text'].tolist()
tamil_data5 = df5['text'].tolist()
tamil_data6 = df6['text'].tolist()
tamil_data = tamil_data1 + tamil_data2 + tamil_data3 + tamil_data4 + tamil_data5 + tamil_data6
english_data = df7 + df['english'].tolist()
text= english_data + tamil_data

Token = Tokenizer(text)
#Token.spm_tokenizer(vocab_size=327875)
Token.Transformers_Tokenizer(vocab_size=400000)

print("Tokenizer Trained Successfully")