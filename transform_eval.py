import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import struct
import argparse
import wandb
import copy
import pandas as pd

from torch.utils.data import DataLoader
from pprint import pprint
from collections import OrderedDict as ordered_dict
from tqdm import tqdm
from ResNet import ResNet18, ResNet34
from torchvision import datasets, transforms

from utils import *

BATCH_SIZE = 128

nrs_range_dict = {
    "IEEE754": (list(range(8, 1, -1)), list(range(23, 1, -1))),
    "Morris": (list(range(4, 1, -1)), list(range(32, 3, -1))),
    "MorrisHEB": (list(range(4, 1, -1)), list(range(32, 3, -1))),
    "MorrisBiasHEB": (list(range(4, 1, -1)), None),
    "Posit": (list(range(4, 0, -1)), list(range(32, 3, -1))),
}


def best_uniform_precision(og_model, dataset_name, nrs_name, wandb_flag, run_name=None):
    """
    Find the best uniform precision with the given NRS
    """

    # get the range of the NRS
    nrs_ranges = nrs_range_dict[nrs_name]
    fst_pos_args = nrs_ranges[0]
    snd_pos_args = nrs_ranges[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    test_dl = get_test_dl(dataset_name)

    results = []

    for a1 in tqdm(fst_pos_args, desc='First argument|exp|g|size'):

        if wandb_flag:
            run = wandb.init(project='Uniform_NRS', name=run_name + "exp|g=" + str(a1), reinit=True)

        for a2 in tqdm(snd_pos_args, desc='Second argument|frac|size'):

            
            print("moe")

            # transform the model from float32 to the given NRS with the given arguments (precision)
            model = copy.deepcopy(og_model)
            model = trf_uniform_precision(model, nrs_name, a1, a2)
            
            acc = eval_model(model, test_dl, device)
            
            if wandb_flag:
                wandb.log({'acc': acc, 'arg_1': a1, 'arg_2': a2})
                # save results
                results.append((a1, a2, acc))
            else:
                print(f'Accuracy: nrs: {nrs_name}, {acc}, arg_1: {a1}, arg_2: {a2}')
            
        if wandb_flag:
            run.finish()
    
    # save to dataframe
    df = pd.DataFrame(results, columns=['arg_1', 'arg_2', 'acc'])
    df.to_csv(f'./results/{run_name}results.csv', index=False)

def main():

    # uniform run
    # python transform_eval.py --arch ResNet18 --trained_model CIFAR100_ResNet18.pth --dataset CIFAR100 --nrs_name IEEE754 --uniform

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ResNet18', help='The architecture of the model')
    parser.add_argument('--trained_model', type=str, default='./model/ResNet18.pth', help='The path of the trained model')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='The dataset to test')
    parser.add_argument('--nrs_name', type=str, default='IEEE754', help='numerical representation system name')
    parser.add_argument('--uniform', action='store_true', help='uniform precision experiment')
    parser.add_argument('--mixed', action='store_true', help='mixed precision experiment')
    parser.add_argument('--wandb', action='store_true', help='wandb experiment')
    args = parser.parse_args()

    if args.dataset == 'CIFAR100':
        num_classes = 100
        # init and load the model
        model_init = args.arch + f'(num_classes={num_classes})'
        model = eval(model_init)
        model.load_state_dict(torch.load(args.trained_model))
        acc = eval_model(model, get_test_dl(args.dataset), torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Accuracy of the original model: {acc}")
        model.to(torch.device('cpu'))

    else:
        print("Only support CIFAR100 dataset")
        raise NotImplementedError

    if args.uniform:
        best_uniform_precision(model, args.dataset, args.nrs_name, args.wandb, f"{args.nrs_name}_")

if __name__ == "__main__":
    main()