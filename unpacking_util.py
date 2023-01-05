import pickle
import dataset_utility

if __name__ == "__main__":
	with open('lyrics.pkl', "rb") as f:
		matrices = []

		# Iterate until there are no more matrix objects in the file
		while True:
			try:
				# Deserialize the next matrix object from the file
				matrix = pickle.load(f)
				
				# Add the matrix to the list
				matrices.append(matrix)
			except EOFError:
				# If there are no more matrix objects in the file, terminate the loop
				break

	datasetUtility = dataset_utility.DatasetUtility()
	matrices = datasetUtility.clean_dataset(matrices)
	
	with open('lyrics_cleaned.txt', 'w') as f:
		for matrix in matrices:
			for lyric in matrix:
				f.write(lyric[0])