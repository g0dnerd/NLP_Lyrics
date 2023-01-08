import genius_parser
import nltk_generation
import markov_generation
import ml_generation
import dataset_utility
import asyncio
import argparse
import csv


async def make_dataset(dataset: str):
    # If dataset mode is specified, read in the list of artists
    with open(args.dataset, 'r') as f:
        artists = f.readlines()
    artists = [artist.strip() for artist in artists]

    # create a parser class and retrieve all song lyrics for the current artist

    # prepare the dataset file
    with open('lyrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lyrics', 'artist'])
    for current_artist in artists:
        lyrics = await fetch_lyrics(current_artist, dumping=True)


async def fetch_lyrics(artist, dumping: bool):
    geniusParser = genius_parser._GeniusParser(artist)
    artist_id = geniusParser.get_artist_id()
    print("Calling API scheduler")
    lyrics = await geniusParser.api_scheduler(dumping)
    lyrics = ' '.join(lyrics)
    return lyrics


def nltk_generate(lyrics: str):
    nltkGenerator = nltk_generation.NltkGenerator()
    new_lyrics = nltkGenerator.generate_lyrics(lyrics)
    print(new_lyrics)


def markov_generate(lyrics: str):
    markovGenerator = markov_generation.MarkovGenerator()
    new_lyrics = markovGenerator.generate_lyrics(lyrics, 10, 10)
    print(new_lyrics)


def train_model():
    datasetUtility = dataset_utility.DatasetUtility()
    lyrics,artists = datasetUtility.unpack_dataset("lyrics.csv")
    lyrics = datasetUtility.clean_dataset(lyrics)
    subsequences = datasetUtility.subsequence_split(lyrics, artists, sub_sequence_length=512)
    bert_classifier = ml_generation.BERTClassifier(batch_size = 32)
    (X_train, y_train), (X_test, y_test) = datasetUtility.split_dataset(subsequences)
    # print(f'"X_train {X_train} has type {type(X_train)}')
    # print(f'"y_train {y_train} has type {type(y_train)}')
    # train_inputs, train_labels, train_attention_mask = datasetUtility.tokenize_dataset(X_train, y_train)
    # test_inputs, test_labels, test_attention_mask = datasetUtility.tokenize_dataset(X_test, y_test)
    # train_dataset = dataset_utility.LyricsDataset(train_inputs, train_labels)
    # test_dataset = dataset_utility.LyricsDataset(test_inputs_test_labels)
    bert_classifier.train(X_train, y_train)
    # output = bert_classifier.model.sequential_forward(X_test)



async def main(artist: str):

    if args.dataset != None:
        # If dataset mode was specified,
        # make a dataset from the artists specified in args.dataset
        await make_dataset(args.dataset)
    elif args.artist != None:
        # If not, fetch lyrics regularly without dumping
        lyrics = await fetch_lyrics(artist, dumping=False)

    # If one of the non-ML generation modes were specified
    if args.mode == "nltk":
        nltk_generate(lyrics)
    elif args.mode == "markov":
        markov_generate(lyrics)
    elif args.mode == "ml":
        # TODO
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
