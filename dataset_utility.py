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

    def tokenize_lyrics(self, lyrics: list, labels: list)->dict:
        # Set a maximum length and truncate the data. BERT can handle max. 512
        max_length = 512

        # Instantiate the tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # Tokenize a batch of lyrics, truncating it to max_length
        tokenized_lyrics = tokenizer.batch_encode_plus(
            lyrics, padding='longest', max_length=max_length, return_tensors="pt")

        # Truncate the list of labels to the same length
        labels = labels[:max_length]
        # Create a mapping from label strings to integers
        label_mapping = {label: i for i, label in enumerate(set(labels))}

        # Convert the labels to integers using the mapping
        labels = [label_mapping[label] for label in labels]
        # Add the labels (artists) to the dictionary
        tokenized_lyrics["labels"] = torch.tensor(labels)

        return tokenized_lyrics

    def split_dataset(self, lyrics, labels, test_size=0.2):
        # Zip the lyrics and labels together
        dataset = list(zip(lyrics, labels))

        # Shuffle the dataset
        random.shuffle(dataset)

        # Calculate the index to split the dataset at
        split_index = int(test_size * len(dataset))

        # Split the dataset into training and test sets
        train_set = dataset[split_index:]
        test_set = dataset[:split_index]

        # Unzip the training and test sets
        X_train, y_train = zip(*train_set)
        X_test, y_test = zip(*test_set)

        return (X_train, y_train), (X_test, y_test)


class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        self.inputs = data["input_ids"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
