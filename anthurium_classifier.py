import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

# Name that this trial will be saved under
trial_name = "transferlearning"
# Name of trial to start from, if using transfer learning
prev_trial_name = "transferlearning"
# Model will train epochs: (start_epoch, n_epochs]
n_epochs = 30
start_epoch = 10
# How often model will save data to csv and plot
epoch_log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
image_size = 256

# Hyperparameters
batch_size_train = 64
batch_size_test = 500
learning_rate = 0.01
momentum = 0.9

# Create result folder
if not os.path.exists(f'results/{trial_name}'):
   os.makedirs(f'results/{trial_name}')


# Transformations
normalization_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.3767436145686272,), (0.1572338936013405,))
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomPerspective(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    normalization_transform
])

# Load in train and test sets
data_path = "anthurium_images_og"
full_dataset = ImageFolder(root=data_path)

train_ratio = 0.8

train_size = int(len(full_dataset) * train_ratio)
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply transformations to train and test datasets
train_dataset.dataset.transform = augment_transform
test_dataset.dataset.transform = normalization_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


def checkpoint(model, filename):
    torch.save(model.state_dict(),filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def train(epoch):
    network.train()
    train_loss = 0
    # Iterate through dataset by batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Transformations are re-applied every time data is loaded, allowing for online augmentation
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset) / batch_size_train
    print(f'Average Train Loss for Epoch {epoch}: {train_loss}')
    train_losses.append(train_loss)


def test():
    network.eval()
    test_loss = 0
    correct = 0
    true_labels = []
    predicted_labels = []
    # Validate test set
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            # Convert tensors to numpy arrays for F1 score calculation
            true_labels.extend(target.numpy())
            predicted_labels.extend(predicted.numpy())
    # Log performance
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy * 100))


def save():
    # Save csv
    dic = {'test_losses': test_losses, 'train_losses': train_losses, 'test_accuracies': test_accuracies}
    df = pd.DataFrame(dic)
    df.to_csv(f'results/{trial_name}/{trial_name}-losses.csv')

    # Save graph
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['test_accuracies'], label='test_accuracies')
    plt.plot(df.index, df['train_losses'], label='train_losses')
    plt.plot(df.index, df['test_losses'], label='test_losses')
    plt.legend()
    plt.grid()
    plt.xlim([0, len(df['test_accuracies']) - 1])
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('Epoch')
    plt.title('Network Performance')
    plt.savefig(f'results/{trial_name}/{trial_name}-performance.pdf', format='pdf')


# Initializations
network = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = network.fc.in_features
network.fc = nn.Linear(num_ftrs, 5)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = [1]
test_losses = []
test_accuracies = []


# Load in previous trained network
if start_epoch > 0:
    resume(network, f"results/{prev_trial_name}/model-epoch-{start_epoch}.pth")
    resume(optimizer, f"results/{prev_trial_name}/optimizer.pth")
    df = pd.read_csv(f'results/{prev_trial_name}/{prev_trial_name}-losses.csv')
    test_losses = list(df['test_losses'])[:start_epoch]
    train_losses = list(df['train_losses'])[:start_epoch + 1]
    test_accuracies = list(df['test_accuracies'])[:start_epoch]
else:
    checkpoint(optimizer, f"results/{prev_trial_name}/optimizer.pth")

# Train and validate while keeping checkpoints
test()
for epoch in range(start_epoch + 1, n_epochs + 1):
    train(epoch)
    test()
    if epoch % epoch_log_interval == 0:
        checkpoint(network, f"results/{trial_name}/model-epoch-{epoch}.pth")
        save()
save()

