
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from torch.utils.data import DataLoader
from ResNet import ResNet18, ResNet34
from tqdm import tqdm

from torchvision import datasets, transforms

import wandb

def main():

    wandb.init(project="resnet18_cifar100")

    EPOCH_NUM = 10
    BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = ResNet18(num_classes=100)
    model = model.to(device)

    dataset = sys.argv[1]

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(root='data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

        test_dataset = datasets.MNIST(root='data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

    elif dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='data', 
                                train=True, 
                                transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                    ]),
                                download=True)

        test_dataset = datasets.CIFAR100(root='data',
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                    ]),
                                    download=True)


    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

    # define loss function
    criterion = nn.CrossEntropyLoss()


    #### Pretty nice hyperparameters - ResNet18 on CIFAR100
    # define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)

    model.train()

    # train model
    for epoch in tqdm(range(EPOCH_NUM), desc='Training'):

        model.train()

        for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                                total=len(train_loader)):
            # get data
            x, y = data, target

            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_hat = model(x)

            # compute loss
            loss = criterion(y_hat, y)

            wandb.log({"train_loss": loss.item()})

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # zero gradients
            optimizer.zero_grad()

        # evaluate model
        model.eval()

        correct = 0
        total = 0
        losses = []

        with torch.no_grad():
            for batch in test_loader:

                # get data
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                # forward pass
                y_hat = model(x)

                # compute loss
                loss = criterion(y_hat, y)

                losses.append(loss.item())

                # get predictions
                _, predictions = torch.max(y_hat, 1)

                # compute accuracy
                correct += (predictions == y).sum().item()
                total += y.shape[0]

        print("Epoch: ", epoch, "Accuracy: ", correct/total)
        wandb.log({"val_accuracy": correct/total})
        wandb.log({"val_loss": sum(losses)/len(losses)})
        wandb.log({"lr": lr_scheduler.get_lr()[0]})

        lr_scheduler.step()


    # save model
    torch.save(model.state_dict(), f"{dataset}_ResNet18.pth")

    wandb.finish()


if __name__ == '__main__':
    main()

"""
Notes: Need augmentation for CIFAR100 to improve accuracy
LR trainer: https://github.com/JosephChenHub/pytorch-lars/blob/7be67c9753ce304f21b5c5ab4fdf736030157fdc/lr_finder.py#L13
"""
