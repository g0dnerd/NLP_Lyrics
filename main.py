import genius_parser
import nltk_generation
import markov_generation
import ml_generation
import dataset_utility
import asyncio
import argparse
import csv


async def make_dataset(file_path: str):
    # If dataset mode is specified, read in the list of artists
    with open(file_path, 'r') as f:
        artists = f.readlines()
    artists = [artist.strip() for artist in artists]

    # prepare the dataset file
    with open('lyrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lyrics', 'artist'])
    for current_artist in artists:
        await fetch_lyrics(current_artist, dumping=True)


async def fetch_lyrics(artist, dumping: bool):
    parser = genius_parser.GeniusParser(artist)
    print("Calling API scheduler")
    lyrics = await parser.api_scheduler(dumping)
    lyrics = ' '.join(lyrics)
    return lyrics


def nltk_generate(lyrics: str):
    nltk_generator = nltk_generation.NltkGenerator()
    new_lyrics = nltk_generator.generate_lyrics(lyrics)
    print(new_lyrics)


def markov_generate(lyrics: str):
    markov_generator = markov_generation.MarkovGenerator()
    new_lyrics = markov_generator.generate_lyrics(lyrics, 10, 10)
    print(new_lyrics)


def train_model():
    lyrics, artists = dataset_utility.unpack_dataset("lyrics.csv")
    lyrics = dataset_utility.clean_dataset(lyrics)
    subsequences = dataset_utility.subsequence_split(lyrics, artists, sub_sequence_length=512)
    bert_classifier = ml_generation.BERTClassifier()
    (X_train, y_train), (X_test, y_test) = dataset_utility.split_dataset(subsequences)
    train_dataset = dataset_utility.tokenize_dataset(X_train, y_train)
    test_dataset = dataset_utility.tokenize_dataset(X_test, y_test)
    training_dataset = dataset_utility.LyricsDataset(train_dataset)
    testing_dataset = dataset_utility.LyricsDataset(test_dataset)

    bert_classifier.train(training_dataset, testing_dataset)


async def main(artist: str):

    if args.dataset is not None:
        # If dataset mode was specified,
        # make a dataset from the artists specified in args.dataset
        await make_dataset(args.dataset)
    elif args.artist is not None:
        # If not, fetch lyrics regularly without dumping
        lyrics = await fetch_lyrics(artist, dumping=False)

        # If one of the non-ML generation modes were specified
        if args.mode == "nltk":
            nltk_generate(lyrics)
        elif args.mode == "markov":
            markov_generate(lyrics)
    elif args.mode == "ml":
        train_model()

if __name__ == "__main__":

    # Parse artist name and generation mode
    parser = argparse.ArgumentParser(
        description='Download song lyrics by an artist and generate more.')
    parser.add_argument(
        '--artist', type=str, help='The artist name to generate lyrics for.\
        Please input this in quotation marks using spaces if necessary.')
    parser.add_argument(
        "--mode", help="The chosen generation mode. Choose between nltk, markov and ml.\
        If ML is specified, it will look for a dataset.csv file")
    parser.add_argument(
        "--dataset", type=str, help="Provide an input list of artists separated by newlines.\
        Does not generate lyrics and instead stores the scraped lyrics to a .csv file.\
        WARNING: an existing file called lyrics.csv will be overwritten. ")
    parser.add_argument(
        "--save", action="store_true", help="Save the scraped lyrics to a file called 'ARTISTNAMElyrics.txt'")
    args = parser.parse_args()

    artist = str(args.artist)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(artist))
