import genius_parser
import nltk_generation
import markov_generation
import asyncio
import argparse

async def main(artist):

    geniusParser = genius_parser._GeniusParser(artist)
    artist_id = geniusParser.get_artist_id()
    songs = geniusParser.get_songs(artist_id)
    lyrics = ""

    await geniusParser.api_scheduler() # call the API scheduler asynchronously

    if args.save:
        with open(args.artist + "lyrics.txt", "w") as f:
            f.write(lyrics)

    if args.mode == "nltk":

        nltkGenerator = nltk_generation.NltkGenerator()
        new_lyrics = nltkGenerator.generate_lyrics(lyrics)

    elif args.mode == "markov":
        markovGenerator = markov_generation.MarkovGenerator()
        new_lyrics = markovGenerator.generate_lyrics(lyrics, 10, 10)


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
    