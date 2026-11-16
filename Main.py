import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms



OVERFITTEST = False



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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.85)

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

# Training the model
num_epochs = 5