import pickle
import re
import unicodedata
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class DatasetUtility:

    def pickle_load_from_file(self, filename):
        with open(filename, "rb") as f:
            matrices = []

            # Iterate until there are no more matrix objects in the file
            while True:
                try:
                    # Deserialize the next matrix object from the file
                    matrix = pickle.load(f)

                    # Add the matrix to the list
                    matrices.append(matrix)
                except EOFError:
                    # If there are no more matrix objects in the file, terminate the loop
                    break
        return matrices

    def clean_dataset(self, dataset):
        for matrix in dataset:
            for lyrics in matrix:
                # Remove everything within square brackets
                lyrics[0] = re.sub(r'\[[^\]]*\]', '', lyrics[0])

                # Remove everything within angle brackets
                lyrics[0] = re.sub(r'\<[^\>]*\>', '', lyrics[0])

                # Remove all newlines
                lyrics[0] = re.sub(r'\n', '', lyrics[0])

                # Remove all stacked empty chars
                lyrics[0] = re.sub(r' {2,}', ' ', lyrics[0])

                # Remove weird Unicode noise
                lyrics[0] = unicodedata.normalize("NFKD", lyrics[0])

        return dataset

    def tokenize_lyrics(self, dataset):
        all_lyrics = [lyric for matrix in dataset for lyric in matrix[0]]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_lyrics)
        sequences = [tokenizer.texts_to_sequences(
            matrix[0]) for matrix in dataset]
        return sequences

    def split_dataset(self, sequences, dataset, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, dataset[1], test_size=test_size)
        return (X_train, y_train), (X_test, y_test)
