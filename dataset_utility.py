import csv
import re
import unicodedata
import torch
import random
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class DatasetUtility:

    def unpack_dataset(self, filename: str)->(list, list):
        with open(filename, "rb") as f:
            lyrics = []
            artists = []
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    lyrics.append(row['lyrics'])
                    artists.append(row['artist'])
        return lyrics, artists

    def clean_dataset(self, lyrics: list)->list:
        for lyric in lyrics:
            # Remove everything within square brackets
            lyric = re.sub(r'\[[^\]]*\]', '', lyric)

            # Remove everything within angle brackets
            lyric = re.sub(r'\<[^\>]*\>', '', lyric)

            # Remove all newl)s
            lyric = re.sub(r'\n', '', lyric)

            # Remove all stacked empty characters
            lyric = re.sub(r' {2,}', ' ', lyric)

            # Remove weird Unicode noise
            lyric = unicodedata.normalize("NFKD", lyric)

        # Remove leading and trailing quotation marks
        for i, s in enumerate(lyrics):
            lyrics[i] = s[1:-1]

        with open('lyrics_cleaned.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for lyric in lyrics:
                writer.writerow([lyric, 'artist'])

        return lyrics

    # def tokenize_dataset(self, X_train, y_train):
        # # Initialize the BERT tokenizer
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 
        # # Tokenize and convert the training data to tensors
        # inputs = tokenizer.batch_encode_plus(X_train, max_length=512, pad_to_max_length=True, return_tensors='pt')
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # labels = torch.tensor(y_train).to(device)
# 
        # return inputs,

    def split_dataset(self, data, train_percentage=0.8):
        random.shuffle(data)
        train_size = int(len(data) * train_percentage)
        train_data = data[:train_size]
        test_data = data[train_size:]
        X_train = [lyric for lyric, artist in train_data]
        y_train = [artist for lyric, artist in train_data]
        X_test = [lyric for lyric, artist in test_data]
        y_test = [artist for lyric, artist in test_data]
        return (X_train, y_train), (X_test, y_test)

    def subsequence_split(self, lyrics: list, artists: list, sub_sequence_length=512)->list:
        sub_sequences = []

        for i in range(len(lyrics)):
            song_lyrics = lyrics[i]
            artist = artists[i]
            for j in range(0, len(song_lyrics), sub_sequence_length):
                sub_sequence = song_lyrics[j:j+sub_sequence_length]
                sub_sequences.append((sub_sequence, artist))
        return sub_sequences


class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.inputs = input_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
