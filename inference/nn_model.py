import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from inference.nn_classifier import Net
from inference.nn_model_utils import train_dataloader, test_dataset, val_dataset

# Define the class-to-label mapping
label_to_class = {0: 'dublin', 1: 'kerry'}
int_to_class = {0: 'Dublin', 1: 'Kerry'}
class_to_int = {'Dublin': 0, 'Kerry': 1}

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

# Load the saved model
model = Net()
model.load_state_dict(torch.load('model_path.pt'))
model = model.to(device)
model.eval()  # set model to evaluation mode

# Initialize test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy of the model: {100 * correct / total}%')

# Get one batch of test data
test_images, test_labels = next(iter(test_dataloader))
test_images, test_labels = images.to(device), test_labels.to(device)

# Run the model on the batch
test_outputs = model(test_images)
_, predicted = torch.max(test_outputs, 1)

# Unnormalize the images for displaying
test_images = test_images / 2 + 0.5
test_images = test_images.cpu().numpy()

# Plot images and model's predictions
fig = plt.figure(figsize=(25, 4))
for idx in range(len(test_images)):  # change 20 to however many images you want to display
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(test_images[idx], (1, 2, 0)))
    title_text = f'Predicted:\n {int_to_class[predicted[idx].item()]}'
    ax.set_title(title_text)
    plt.subplots_adjust(hspace = 0.5)  # adjust space between plots

plt.show()
