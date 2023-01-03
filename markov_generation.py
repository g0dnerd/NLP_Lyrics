import re
import random


class MarkovGenerator:

    def generate_lyrics(self, corpus, max_lines, max_words_per_line):
        # Split the corpus into a list of words
        words = re.findall(r'\w+', corpus)

        # Create a dictionary that maps words to lists of words that follow them
        markov_chain = {}
        for i, word in enumerate(words[:-1]):
            if word not in markov_chain:
                markov_chain[word] = []
            markov_chain[word].append(words[i + 1])

        # Generate random lyrics
        lyrics = ""
        line_number = 1
        word = random.choice(list(markov_chain.keys()))
        line = word.capitalize()
        num_words = 1
        while line_number <= max_lines:
            if word in markov_chain and num_words < max_words_per_line:
                # Choose the next word according to the probabilities in the markov chain
                next_word_probs = markov_chain[word]
                weights = [next_word_probs.count(w) for w in next_word_probs]
                next_word = random.choices(next_word_probs, weights=weights)[0]
                line += ' ' + next_word
                word = next_word
                num_words += 1
            else:
                # End the line and start a new one
                lyrics += line + "\n"
                line_number += 1
                if line_number > max_lines:
                    break
                word = random.choice(list(markov_chain.keys()))
                line = word.capitalize()
                num_words = 1
        return lyrics
