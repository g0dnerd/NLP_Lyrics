import transformers
import torch

class TransformerGenerator:
  
  def train_on_dataset(self, dataset):
    # Load the BERT model
    model = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Tokenize the input
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    print(dataset)
    # Define a mapping from artists to integer labels
    all_artists = [artist for song_lyrics in dataset for artist in song_lyrics[1]]
    artist_to_id = {artist: i for i, artist in enumerate(set(all_artists))}

    # Define a training loop
    for song_lyrics in dataset:
      lyrics, artists = song_lyrics
      input_ids = tokenizer.encode(lyrics, return_tensors='pt')
      labels = torch.tensor([artist_to_id[artist] for artist in artists], dtype=torch.long)

      # Calculate the loss and backpropagate the gradients
      loss = model(input_ids, labels=labels)[0]
      loss.backward()

      # Update the model weights
      optimizer.step()
      model.zero_grad()

