import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader


class BERTClassifier:
    def __init__(self, train_dataset, test_dataset, batch_size, epochs=4):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
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

    def train(self):
        # Set the model to training mode
        self.model.train()

        # Use the DataLoader class to create an iterator over the training data
        train_data_iterator = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Loop over the training data for the specified number of epochs
        for epoch in range(self.epochs):
            print("Epoch:", epoch+1)

            # Loop over the training data in the current epoch
            for inputs, labels in train_data_iterator:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs, labels=labels)
                loss = self.loss_fn(outputs[0], labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()