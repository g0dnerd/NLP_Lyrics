import re
import requests
import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup


class _GeniusParser:

    def __init__(self, artist):

        # Your Genius API access token
        self.access_token = 'pXlD53LSYo8uToC1rJ7Y5IRhMpxhAP7xBZag_4wgQ-8uZm2kBLkYQGlYFwpuPay0'

        # The base URL for the Genius API
        self.base_url = 'https://api.genius.com'

        # Set the headers
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

        # The Genius API endpoint for getting an artist by name
        self.artist_endpoint = '/search'

        self.artist = artist

    def get_artist_id(self):
        """
        Gets the Genius ID for an artist by name.
        """

        print("Getting Genius artist ID")

        params = {'q': self.artist}
        response = requests.get(self.base_url + self.artist_endpoint,
                                params=params, headers=self.headers)
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
            # Make the API request
            response = requests.get(
                self.base_url + artist_songs_endpoint,
                params=params, headers=self.headers)
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

    async def get_song_urls(self):
        """
        Gets the URL for all songs by an artist on Genius.com.
        """

        song_urls = []

        # Set the search parameters
        params = {
            "q": self.artist, 'per_page': 50
        }

        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(self.base_url + self.artist_endpoint, params=params, headers=self.headers) as response:
                    api_response = await response.json()

                # add the URLs of the matching songs to the song_urls list

                for song in api_response['response']['hits']:
                    song_url = song['result']['url']
                    song_urls.append(song_url)
                    # check if there are more pages of results
                    if api_response['response']['next_page'] is None:
                        break
                    else:
                        params['page'] = api_response['response']['next_page']
        return song_urls

        # # Send the request and get the response
        # response = requests.get(self.base_url + self.artist_endpoint, params=params, headers=self.headers)
#
        # # Get the JSON data from the response
        # data = response.json()
#
        # # Get the first hit from the search results
        # first_hit = data['response']['hits'][0]
#
        # # Get the URL of the lyrics page from the first hit
        # song_url = first_hit['result']['url']

        # return song_url

    async def get_lyrics(self, session, url):

        print(f'Retrieving lyrics from URL: {url}')
        
        async with session.get(url) as response:
            html = await response.text()

        # Make a request to the URL
        # page = requests.get(url)

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
        lyrics = re.sub(r'\[.+?\]', '', lyrics)
        # lyrics = re.sub(r'You might also like.*', '', lyrics)
        # lyrics = re.sub(r'Embed', '', lyrics)

        # Re-insert new lines.
        lyrics = re.sub('<br>|<br/>', '\n', lyrics)

        return lyrics

    async def download_lyrics(song_urls):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in song_urls:
                task = asyncio.ensure_future(get_lyrics(session, url))
                tasks.append(task)
                await asyncio.sleep(1)  # delay each request by 1 second
            lyrics = await asyncio.gather(*tasks)
            print(f'Number of lyrics: {len(lyrics)}')
        return lyrics

    async def api_scheduler(self):
        song_urls = await self.get_song_urls()
        lyrics = await self.download_lyrics(song_urls)
