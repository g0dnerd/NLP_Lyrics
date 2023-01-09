import re
import requests
import os
import aiohttp
import asyncio
import pickle
import csv
from bs4 import BeautifulSoup


class GeniusParser:

    def __init__(self, artist):

        # Your Genius API access token
        self.access_token =\
            'pXlD53LSYo8uToC1rJ7Y5IRhMpxhAP7xBZag_4wgQ-8uZm2kBLkYQGlYFwpuPay0'

        # The base URL for the Genius API
        self.base_url = 'https://api.genius.com'

        # Set the headers
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

        # The Genius API endpoint for getting an artist by name
        self.artist_endpoint = '/search'

        self.artist = artist

        self.artist_id = 0

    def get_artist_id(self):
        """
        Gets the Genius ID for an artist by name.
        """

        print(f'Getting Genius artist ID for {self.artist}')

        params = {'q': self.artist}
        response = requests.get(self.base_url + self.artist_endpoint,
                                params=params, headers=self.headers)
        data = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            # Get the first artist in the list of artists
            artist = data['response']['hits'][0]['result']['primary_artist']
            # Return the artist's Genius ID
            self.artist_id = artist['id']
        else:
            # If the request was not successful,
            # print the error message and return None
            print(data['error'])

    async def get_song_urls(self):
        """
        Gets the URL for all songs by an artist on Genius.com.
        """

        song_urls = []

        # The current page of results
        page = 1
        # A flag indicating whether there are more songs to retrieve
        more_songs = True

        artist_songs_endpoint = '/artists/' + str(self.artist_id) + '/songs'

        async with aiohttp.ClientSession() as session:
            while more_songs:

                # Set the search parameters
                params = {'page': page}
                async with session.get(self.base_url + artist_songs_endpoint,
                                       params=params, headers=self.headers) as response:
                    api_response = await response.json()

                # add the URLs of the matching songs to the song_urls list

                # Check if the request was successful
                if response.status == 200:
                    for song in api_response['response']['songs']:
                        song_url = song['url']
                        # sprint(f"successfully retrieved {song_url}")
                        song_urls.append(song_url)
                        more_songs = api_response['response']['next_page'] != None
                    page += 1
                else:
                    more_songs = False
                    print(data['error'])
        print(f'Found %d song URLs' % len(song_urls))
        return song_urls

    async def get_lyrics(self, session, url: str)->str:

        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
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
        lyrics = re.sub(r'\<.+?\>', '', lyrics)

        # Re-insert new lines.
        lyrics = re.sub('<br>|<br/>', '\n', lyrics)
        lyrics = re.sub(r'\n', '', lyrics)
        lyrics = re.sub(r'\s{2,}', ' ', lyrics)

        return lyrics

    async def download_lyrics(self, song_urls: list, dumping: bool):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in song_urls:
                task = asyncio.ensure_future(self.get_lyrics(session, url))
                tasks.append(task)
                # await asyncio.sleep(1)  # delay each request by 1 second
            lyrics = await asyncio.gather(*tasks)
            # If dumping to dataset was specified, write lyrics to a csv. file
            if dumping:
                with open('lyrics.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    rows = [[lyric, self.artist] for lyric in lyrics]
                    writer.writerows(rows)
        return lyrics

    async def api_scheduler(self, dumping: bool):
        print("Retrieving song URLs")
        song_urls = await self.get_song_urls()
        print("Scraping lyrics from HTML")
        lyrics = await self.download_lyrics(song_urls, dumping)
        # print(f'Finished API calls returned %d lyrics' %len(lyrics))
        return lyrics
