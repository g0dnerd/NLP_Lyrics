import genius_parser
import nltk_generation
import ml_generation
import markov_generation
import asyncio
import argparse
import re

async def main(artist):

    geniusParser = genius_parser._GeniusParser(artist)
    artist_id = geniusParser.get_artist_id()
    songs = geniusParser.get_songs(artist_id)
    lyrics = ""

    print("Calling API scheduler")

    lyrics = await geniusParser.api_scheduler() # call the API scheduler asynchronously

    lyrics_string = '\n'.join(lyrics)
    lyrics_string = re.sub(r'\n{2,}', '\n', lyrics_string)
    lyrics_string = re.sub(r' {2,}', '', lyrics_string)

    if args.save:
        with open(args.artist + "lyrics.txt", "w") as f:
            f.write(lyrics_string)

    if args.mode == "nltk":

        nltkGenerator = nltk_generation.NltkGenerator()
        new_lyrics = nltkGenerator.generate_lyrics(lyrics_string)

    elif args.mode == "markov":
        markovGenerator = markov_generation.MarkovGenerator()
        new_lyrics = markovGenerator.generate_lyrics(lyrics_string, 10, 10)

    elif args.mode == "ml":
        mlGenerator = ml_generation.MlGenerator()
        processed_lyrics = [mlGenerator.clean_lyrics(lyric) for lyric in lyrics]
        mlGenerator.extract_features(processed_lyrics)



    print(new_lyrics)


if __name__ == "__main__":

    # Parse artist name and generation mode
    parser = argparse.ArgumentParser(
        description='Download song lyrics by an artist and generate more.')
    parser.add_argument(
        'artist', type=str, help='The artist name to generate lyrics for. Please input this in quotation marks using spaces if necessary.')
    parser.add_argument(
        "--mode", help="The chosen generation mode. nltk and markov currently exist.")
    parser.add_argument(
        "--save", action="store_true", help="Save the scraped lyrics to a file called 'ARTISTNAMElyrics.txt'")
    args = parser.parse_args()

    artist = str(args.artist)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(artist))
    