import torch
import transformers
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader


class BERTClassifier:
    def __init__(self, epochs=4, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load the BERT model
        self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(self.device)

        # Create the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        # Total number of training steps
        self.total_steps = 0

        # Create the learning rate scheduler
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                                      num_training_steps=self.total_steps)

    def pad_text(self, input_ids):
        # Pad the text
        while len(input_ids) < 64:
            input_ids.append(0)
        return input_ids

    def train_epoch(self, train_dataloader):
        self.model.train()

        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                print(f"Batch {step}/{len(train_dataloader)}")

            # Unpack the batch
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            # Backward pass
            loss.backward()

            # Update the weights
            self.optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")

    def evaluate(self, test_dataloader):
        self.model.eval()

        total_loss = 0
        total_accuracy = 0

        for batch in test_dataloader:
            # Unpack the batch
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]

            total_loss += loss.item()

            # Calculate the accuracy
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_accuracy += self.calculate_accuracy(logits, label_ids)

        avg_val_accuracy = total_accuracy / len(test_dataloader)
        print(f"Accuracy: {avg_val_accuracy}")

        avg_val_loss = total_loss / len(test_dataloader)
        print(f"Validation loss: {avg_val_loss}")

    def calculate_accuracy(self, logits, labels):
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def predict(self, text):
        # Tokenize the text
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Pad the text
        input_ids = self.pad_text(input_ids)

        # Convert the input to a tensor
        input_ids = torch.tensor(input_ids)

        # Add a batch dimension
        input_ids = input_ids.unsqueeze(0)

        # Put the model in evaluation mode
        self.model.eval()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs[0]

        # Get the predictions
        predictions = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        return predictions

    def train(self, train_dataset, test_dataset):
        # Create the dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.total_steps = len(train_dataloader) * self.epochs

        # Train the model
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("-" * 10)

            self.train_epoch(train_dataloader)
            self.evaluate(test_dataloader)
