import genius_parser
import nltk_generation
import markov_generation
import asyncio
import argparse
import re

async def main(artist):

    if args.train == None:
        if args.dataset != None:

            # If dataset mode is specified, read in the list of artists
            with open(args.dataset, 'r') as f:
                artists = f.readlines()
            artists = [artist.strip() for artist in artists]
            
            # create a parser class and retrieve all songs for the current artist
            for current_artist in artists:
                geniusParser = genius_parser._GeniusParser(current_artist)
                artist_id = geniusParser.get_artist_id()
                # songs = geniusParser.get_songs(artist_id)
                lyrics = ""

                print("Calling API scheduler")

                # call the API scheduler asynchronously

                lyrics = await geniusParser.api_scheduler(True)
        else:
            geniusParser = genius_parser._GeniusParser(artist)
            artist_id = geniusParser.get_artist_id()
            songs = geniusParser.get_songs(artist_id)
            lyrics = await geniusParser.api_scheduler(False)

            print("Calling API scheduler")

        lyrics_string = '\n'.join(lyrics)
        lyrics_string = re.sub(r'\n{2,}', '\n', lyrics_string)
        lyrics_string = re.sub(r' {2,}', '', lyrics_string)

    else:
        datasetUtility = dataset_utility.DatasetUtility()
        dataset = datasetUtility.pickle_load_from_file(args.train)
        dataset = datasetUtility.clean_dataset(dataset)
        sequences = datasetUtility.tokenize_lyrics(dataset)
        (X_train, y_train), (X_test, y_test) = datasetUtility.split_dataset(sequences, dataset)

    if args.mode == "nltk":
        nltkGenerator = nltk_generation.NltkGenerator()
        new_lyrics = nltkGenerator.generate_lyrics(lyrics_string)
        print(new_lyrics)

    elif args.mode == "markov":
        markovGenerator = markov_generation.MarkovGenerator()
        new_lyrics = markovGenerator.generate_lyrics(lyrics_string, 10, 10)
        print(new_lyrics)

    elif args.mode == "ml":
        mlGenerator = ml_generation.MlGenerator()
        processed_lyrics = [mlGenerator.clean_lyrics(
            lyric) for lyric in lyrics]
        mlGenerator.extract_features(processed_lyrics)
        print(new_lyrics)


if __name__ == "__main__":

    # Parse artist name and generation mode
    parser = argparse.ArgumentParser(
        description='Download song lyrics by an artist and generate more.')
    parser.add_argument(
        '--artist', type=str, help='The artist name to generate lyrics for. Please input this in quotation marks using spaces if necessary.')
    parser.add_argument(
        "--mode", help="The chosen generation mode. nltk and markov currently exist.")
    parser.add_argument(
        "--dataset", type=str, help="Provide an input list of artists separated by newlines. Does not generate lyrics and instead stores the scraped lyrics to a pickle.")
    parser.add_argument(
        "--save", action="store_true", help="Save the scraped lyrics to a file called 'ARTISTNAMElyrics.txt'")
    parser.add_argument(
        "--train", help="Specify the filename for the dataset to train on.")
    args = parser.parse_args()

    artist = str(args.artist)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(artist))
