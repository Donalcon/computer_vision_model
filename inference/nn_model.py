import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch

from inference.nn_classifier import Net
from inference.nn_model_utils import train_dataloader, test_dataset, val_dataset

# Create an instance of the model
model = Net()

# Use a GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#-------------------------------
# Number of epochs (iterations over the entire dataset)
epochs = 10

for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        # Move data and labels to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'model_path.pt')
