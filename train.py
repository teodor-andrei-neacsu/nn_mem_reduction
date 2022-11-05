
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from ResNet import ResNet18
from tqdm import tqdm

from torchvision import datasets, transforms


def main():

    BATCH_SIZE = 128

    # load model
    model = ResNet18()

    train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

    test_dataset = datasets.MNIST(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    # train model
    for epoch in tqdm(range(4)):

        model.train()

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get data
            x, y = data, target

            # forward pass
            y_hat = model(x)

            # compute loss
            loss = criterion(y_hat, y)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # zero gradients
            optimizer.zero_grad()

            # print("Loss: ", loss.item())

        # evaluate model
        model.eval()

        correct = 0
        total = 0

        for batch in test_loader:

            # get data
            x, y = batch

            # forward pass
            y_hat = model(x)

            # compute loss
            loss = criterion(y_hat, y)

            # get predictions
            _, predictions = torch.max(y_hat, 1)

            # compute accuracy
            correct += (predictions == y).sum().item()
            total += y.shape[0]

        print("Epoch: ", epoch, "Accuracy: ", correct/total)

    # save model
    torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    main()