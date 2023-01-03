import genius_parser
import generate_lyrics
import argparse

if __name__ == "__main__":

    # Parse artist name
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", help="Artist name")
    args = parser.parse_args()

    geniusParser = genius_parser._GeniusParser()
    artist_id = geniusParser.get_artist_id({args.artist})
    songs = geniusParser.get_songs(artist_id)
    lyrics = ""
    print("Scraping lyrics from html")
    
    for song in songs:
        url = geniusParser.get_song_url(
            song['primary_artist']['name'], song['title'])
        lyrics += geniusParser.get_lyrics(url)

    lyricsGenerator = generate_lyrics.LyricsGenerator()
    new_lyrics = lyricsGenerator.generate_lyrics(lyrics)

    for line in new_lyrics: 
        print(line)
