import torch
import transformers
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader


class BERTClassifier:
    def __init__(self, batch_size, epochs=4):
        self.batch_size = batch_size
        self.epochs = epochs

        # Load the BERT model
        self.model = transformers.BertForSequenceClassification.\
            from_pretrained("bert-base-cased")

        # Use Adam as the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        # Set the device (CPU or GPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define the loss function and the optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, X_train, y_train):

        # Initialize the BERT tokenizer and a Label Encoder
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        label_encoder = LabelEncoder()

        # Tokenize and convert the training data to tensors
        inputs = tokenizer.batch_encode_plus(
            X_train, max_length=512, padding='max_length', return_tensors='pt')
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        integer_labels = label_encoder.fit_transform(y_train)
        labels = torch.tensor(integer_labels)

        # Set the model to training mode
        self.model.train()

        # Loop over the training data for the specified number of epochs
        for epoch in range(self.epochs):
            for i, (inputs, labels) in enumerate(train_dataloader):
                # Move inputs and labels to the device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
