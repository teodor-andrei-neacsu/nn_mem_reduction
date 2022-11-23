import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import struct
import numpy as np
import multiprocessing

from tqdm import tqdm
from collections import OrderedDict as ordered_dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from functools import partial

BATCH_SIZE = 128

def get_test_dl(dataset_name) -> DataLoader:
    """
    Get the test dataloader of the given dataset
    """
    if dataset_name == 'CIFAR100':
        test_dataset = datasets.CIFAR100(root='data', 
                                    train=False, 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                    ]),
                                    download=True)

        test_loader = DataLoader(dataset=test_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False)

    else:
        print('Dataset not supported')
        raise NotImplementedError


    return test_loader


@torch.no_grad()
def eval_model(model, test_loader, device):
    """
    Evaluate the model return the accuracy.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating model'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def get_layer_group_dict(model):
    """
    Get the layer group dict of the model
    """
    layer_group_dict = ordered_dict()
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_group_dict:
            layer_group_dict[layer_name] = []
        layer_group_dict[layer_name].append(name)
    return layer_group_dict


def convert_layer_weigths(name, param, nrs_name, nrs_a1, nrs_a2):


    # save the layer as npy
    np.save(f'./weight_trf/{name}.npy', param.data.numpy().astype(">f"))

    # transform the layer numerical representation system
    subprocess.check_call(['java', '-Xmx1024m', '-jar', 'convert_weights.jar', f'./weight_trf/{name}.npy', f'./weight_prc/{name}.npy', nrs_name, str(nrs_a1), str(nrs_a2)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)

    
def trf_uniform_precision(model, nrs_name, nrs_a1, nrs_a2) -> torch.nn.Module:
    """
    Transform the layers to uniform precision with the given NRS
    """

    sd = model.state_dict()

    mod_layer = None
    mod_layer_name = None

    # create argument list for multiprocessing
    arg_list = []
    for name, param in model.named_parameters():
        arg_list.append((name, param, nrs_name, nrs_a1, nrs_a2))

    print(' Transforming the model to uniform precision with NRS: {} {} {}'.format(nrs_name, nrs_a1, nrs_a2))

    # compute the layer group dict
    pool = multiprocessing.Pool(processes=4)
    pool.starmap(convert_layer_weigths, arg_list)

    # sequential version
    # for args in tqdm(arg_list, desc='Transforming model'):
    #     print(args[0])
    #     convert_layer_weigths(*args)

    for name, param in model.named_parameters():
        # load the modified layer
        mod_layer = np.load(f'./weight_prc/{name}.npy').astype("<f")
        sd[name] = torch.tensor(mod_layer)

    model.load_state_dict(sd)
    return model


def main():
    pass

if __name__ == '__main__':
    main()