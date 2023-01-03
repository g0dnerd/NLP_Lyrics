import re
import requests
import os
from bs4 import BeautifulSoup


class _GeniusParser:

    # Your Genius API access token
    access_token = 'pXlD53LSYo8uToC1rJ7Y5IRhMpxhAP7xBZag_4wgQ-8uZm2kBLkYQGlYFwpuPay0'

    # The base URL for the Genius API
    base_url = 'https://api.genius.com'

    # The Genius API endpoint for getting an artist by name
    artist_endpoint = '/search'

    def get_artist_id(self, artist_name):
        """
        Gets the Genius ID for an artist by name.
        """

        print("Getting Genius artist ID")

        params = {'q': artist_name}
        headers = {'Authorization': 'Bearer ' + self.access_token}
        response = requests.get(self.base_url + self.artist_endpoint,
                                params=params, headers=headers)
        data = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            # Get the first artist in the list of artists
            artist = data['response']['hits'][0]['result']['primary_artist']
            # Return the artist's Genius ID
            return artist['id']
        else:
            # If the request was not successful,
            # print the error message and return None
            print(data['error'])
            return None

    def get_songs(self, artist_id):
        """
        Gets a list of all songs by an artist.
        """
        # The list of songs
        songs = []
        # The current page of results
        page = 1
        # A flag indicating whether there are more songs to retrieve
        more_songs = True

        print("Retrieving all songs by artist")

        # Iterate until all songs have been retrieved
        while more_songs:
            # The Genius API endpoint for getting an artist's songs
            artist_songs_endpoint = '/artists/{id}/songs'.format(id=artist_id)
            # The API request parameters
            params = {'page': page}
            # The API request headers
            headers = {'Authorization': 'Bearer ' + self.access_token}
            # Make the API request
            response = requests.get(
                self.base_url + artist_songs_endpoint,
                params=params, headers=headers)
            # Get the API response data
            data = response.json()

            # Check if the request was successful
            if response.status_code == 200:
                # Get the list of songs
                song_list = data['response']['songs']
                # Add the songs to the list
                songs.extend(song_list)
                # Check if there are more songs
                more_songs = data['response']['next_page'] != None
                # Increment the page number
                page += 1
            else:
                # If the request was not successful,
                # set the flag to False and print the error message
                more_songs = False
                print(data['error'])

        # Return the list of songs
        return songs

    def get_song_url(self, artist_name, song_name):
        """
        Gets the URL for a song on Genius.com.
        """
        
        # Set the headers
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        # Set the search parameters
        params = {
            "q": f"{song_name} {artist_name}"
        }

        # Send the request and get the response
        response = requests.get(self.base_url + self.artist_endpoint, params=params, headers=headers)

        # Get the JSON data from the response
        data = response.json()

        # Get the first hit from the search results
        first_hit = data['response']['hits'][0]

        # Get the URL of the lyrics page from the first hit
        song_url = first_hit['result']['url']

        return song_url

    def get_lyrics(self, url):

        # Make a request to the URL
        page = requests.get(url)

        # Reformat the HTML and add indentation
        soup = BeautifulSoup(page.content, 'html.parser')
        html = soup.prettify()

        # Parse the reformatted HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Find the first div element with the
        # class attribute 'data-lyrics-container'
        container_div = soup.findAll("div", {"data-lyrics-container": "true"})

        # Remove all div tags from the raw text
        tag_regex = re.compile("</?(span|div|a)[^>]*>")

        # If the song has no text, add an empty string
        try:
            lyrics = re.sub(tag_regex, '', str(container_div[0]))
        except IndexError:
            lyrics = ""

        # Remove comments and other noise
        lyrics = re.sub(r'\[.*\]', '', lyrics)
        lyrics = re.sub(r'You might also like.*', '', lyrics)
        lyrics = re.sub(r'Embed', '', lyrics)

        # Re-insert new lines.
        lyrics = re.sub('<br>|<br/>', '\n', lyrics)

        return lyrics
