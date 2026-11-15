import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms

# Datasets for Training, Validation, and Testing.
fulltrain_dataset = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
train_dataset, val_dataset = random_split(fulltrain_dataset, [50000, 10000])

print(test_dataset[0][0].shape)