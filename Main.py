import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import torchmetrics



OVERFITTEST = False
num_epochs = 10
learning_rate = 0.01
momentum_set = 0.9


# Datasets for Training, Validation, and Testing.
fulltrain_dataset = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
train_dataset, val_dataset = random_split(fulltrain_dataset, [50000, 10000], generator=torch.Generator().manual_seed(20))

#Dataloaders for Training, Validation, and Testing.
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Neural Network Model
torch.manual_seed(30)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

# loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum_set)

# Overfit Test
if (OVERFITTEST == True):
    image, label = next(iter(train_loader))
    for epoch in range(500):
        model.train()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch % 50 == 0):
            print(loss.item())

# Training the model.

epoch = 0
train_loss_list = []
val_loss_list = []
metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

for epoch in range(num_epochs):
    epoch = epoch + 1
    # Training Loop
    model.train()
    training_loss = 0.0
    for images, labels in train_loader:
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_loss += loss.item()

    train_loss = training_loss / len(train_loader)
    train_loss_list.append(train_loss)

    # Validation Loop
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = criterion(output, labels)
            validation_loss += loss.item()
            metric.update(output, labels)
    val_loss = validation_loss / len(val_loader)
    val_loss_list.append(val_loss)
    val_accuracy = metric.compute()
    metric.reset()

    print(f"epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")

#Graphing the training and validation loss
epochs = range(1, num_epochs + 1)


plt.plot(epochs, train_loss_list, label='Training Loss')
plt.plot(epochs, val_loss_list, label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()