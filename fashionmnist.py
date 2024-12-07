import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cnn import *
from ffn import *

'''
In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.
'''

# I was getting a lot of errors when trying to do the plotting, so I added exception handling and printing out error
# messages for my debugging. This made me consolidate my code into a few methods


# PART 1: Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

batch_size = 64

def plot_prediction_examples(model, loader, model_name):
    try:
        # Get a batch of test images
        dataiter = iter(loader)
        images, labels = next(dataiter)

        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        correct_idx = None
        incorrect_idx = None

        for idx in range(len(labels)):
            pred = predicted[idx]
            true_label = labels[idx]

            if pred == true_label and correct_idx is None:
                correct_idx = idx
            elif pred != true_label and incorrect_idx is None:
                incorrect_idx = idx

            if correct_idx is not None and incorrect_idx is not None:
                break

        plt.figure(figsize=(10, 5))

        # Plot correct prediction
        plt.subplot(1, 2, 1)
        img = images[correct_idx].squeeze().cpu().numpy()
        img = (img * 0.5 + 0.5)  # unnormalize
        plt.imshow(img, cmap='gray')
        plt.title(f'Correct\nPred: {classes[predicted[correct_idx]]}\nTrue: {classes[labels[correct_idx]]}')
        plt.axis('off')

        # Plot incorrect prediction
        plt.subplot(1, 2, 2)
        img = images[incorrect_idx].squeeze().cpu().numpy()
        img = (img * 0.5 + 0.5)  # unnormalize
        plt.imshow(img, cmap='gray')
        plt.title(f'Incorrect\nPred: {classes[predicted[incorrect_idx]]}\nTrue: {classes[labels[incorrect_idx]]}')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting predictions for {model_name}: {str(e)}")


def plot_losses(losses, num_epochs, model_name):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), losses)
        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{model_name}_loss.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {model_name}: {str(e)}")


def train_models():
    # PART 2: Load the dataset
    trainset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    testset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # PART 3: Initialize models
    feedforward_net = FF_Net()
    conv_net = Conv_Net()

    # PART 4: Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
    optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

    ffn_losses = []
    cnn_losses = []

    # PART 5: Training FFN
    num_epochs_ffn = 15
    print("Training FFN...")
    for epoch in range(num_epochs_ffn):
        running_loss_ffn = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer_ffn.zero_grad()
            outputs = feedforward_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ffn.step()
            running_loss_ffn += loss.item()

        epoch_loss = running_loss_ffn / len(trainloader)
        ffn_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, FFN Training loss: {epoch_loss:.4f}")

    print('Finished Training FFN')
    torch.save(feedforward_net.state_dict(), 'ffn.pth')

    # Training CNN
    num_epochs_cnn = 10
    print("Training CNN...")
    for epoch in range(num_epochs_cnn):
        running_loss_cnn = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer_cnn.zero_grad()
            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cnn.step()
            running_loss_cnn += loss.item()

        epoch_loss = running_loss_cnn / len(trainloader)
        cnn_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, CNN Training loss: {epoch_loss:.4f}")

    print('Finished Training CNN')
    torch.save(conv_net.state_dict(), 'cnn.pth')

    # PART 6: Evaluation
    correct_ffn = 0
    total_ffn = 0
    correct_cnn = 0
    total_cnn = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            # Evaluate FFN
            outputs_ffn = feedforward_net(images)
            _, predicted_ffn = torch.max(outputs_ffn.data, 1)
            total_ffn += labels.size(0)
            correct_ffn += (predicted_ffn == labels).sum().item()

            # Evaluate CNN
            outputs_cnn = conv_net(images)
            _, predicted_cnn = torch.max(outputs_cnn.data, 1)
            total_cnn += labels.size(0)
            correct_cnn += (predicted_cnn == labels).sum().item()

    print('Accuracy for feedforward network: ', correct_ffn / total_ffn)
    print('Accuracy for convolutional network: ', correct_cnn / total_cnn)

    # PART 7: Plotting
    try:
        # Plot losses
        plot_losses(ffn_losses, num_epochs_ffn, 'ffn')
        plot_losses(cnn_losses, num_epochs_cnn, 'cnn')

        # Plot predictions
        plot_prediction_examples(feedforward_net, testloader, 'ffn')
        plot_prediction_examples(conv_net, testloader, 'cnn')
    except Exception as e:
        print(f"Error in plotting: {str(e)}")


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        train_models()
    except Exception as e:
        print(f"Error in main: {str(e)}")