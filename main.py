import genius_parser
import nltk_generation
import markov_generation
import argparse

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

    geniusParser = genius_parser._GeniusParser()
    artist_id = geniusParser.get_artist_id({args.artist})
    songs = geniusParser.get_songs(artist_id)
    print("Parsed {} songs by {}".format(len(songs), args.artist))
    lyrics = ""

    song_counter = 1
    for song in songs:
        lyrics = geniusParser.api_scheduler(songs)
        # url = geniusParser.get_song_url(
            # song['primary_artist']['name'], song['title'])
        # lyrics += geniusParser.get_lyrics(url)
        # print("Scraping lyrics for song %d out of %d" %(song_counter, len(songs)), end='\r')
        # song_counter += 1

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
