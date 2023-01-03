import os
import nltk
import random
import sys


class NltkGenerator:
    def generate_lyrics(self, lyrics):

        print("Generating new lyrics")

        # Read in the input and split it into individual lines
        lines = lyrics.split("\n")

        # Use NLTK's built-in punkt tokenizer
        # to split the lines into individual words
        words = []
        for line in lines:
            words.extend(nltk.word_tokenize(line))

        # Create a list of tuples, where each tuple
        # consists of two adjacent words in the input
        pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]

        # Create a dictionary that maps each word in the input
        # to a list of words that follow it
        word_dict = {}
        for word_1, word_2 in pairs:
            if word_1 in word_dict.keys():
                word_dict[word_1].append(word_2)
            else:
                word_dict[word_1] = [word_2]

        # Use the dictionary to generate new lyrics
        num_lines = random.randint(5, 10)  # Generate a random number of lines
        new_lyrics = []
        for i in range(num_lines):
            # Choose a random starting word
            word = random.choice(list(word_dict.keys()))
            line = []
            # Keep adding words until we reach the end of the line
            while word[-1] not in [".", "!", "?"]:
                line.append(word)
                word = random.choice(word_dict[word])
            line.append(word)
            new_lyrics.append(" ".join(line))

        return new_lyrics
