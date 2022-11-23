
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from torch.utils.data import DataLoader
from ResNet import ResNet18
from tqdm import tqdm

from torchvision import datasets, transforms

def main():

    # get dataset argument
    dataset = sys.argv[1]

    BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = ResNet18()
    model.load_state_dict(torch.load('model_prc.pth'))
    model.to(device)

    if dataset == "MNIST":
        # load test data
        test_dataset = datasets.MNIST(root='data',
                                    train=False,
                                    transform=transforms.ToTensor())

    elif dataset == "CIFAR100":
        # load and normalize CIFAR100 test data
        test_dataset = datasets.CIFAR100(root='data', 
                                    train=False, 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                    ]),
                                    download=True)
    else:
        print("Invalid dataset")
        return

    test_loader = DataLoader(dataset=test_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False)

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {(100 * correct / total)}%')



if __name__ == "__main__":

    main()