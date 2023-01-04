import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import argparse

class MlGenerator:

	def clean_lyrics(self, lyrics):
	    # Convert lyrics to lowercase
	    lyrics = lyrics.lower()

	    # Remove newline characters
	    lyrics = lyrics.replace('\n', ' ')

	    # Remove punctuation
	    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))

	    # Remove numbers
	    lyrics = re.sub(r'\d+', '', lyrics)

	    # Remove stop words
	    stop_words = set(stopwords.words('english'))
	    words = lyrics.split()
	    lyrics = [word for word in words if word not in stop_words]

	    # Join words to form processed lyrics
	    lyrics = " ".join(lyrics)

	    return lyrics

	def extract_features(self, lyrics):
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(lyrics)
		print(X)

	def train_model(self, dataset):
		
		# Create a logistic regression model
		model = LogisticRegression()

		# Split the data into training and test sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
