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
    "MorrisBiasHEB": (list(range(4, 1, -1)), list(range(32, 3, -1))),
    "MorrisUnaryHEB": (None, list(range(32, 3, -1))),
    "Posit": (list(range(4, -1, -1)), list(range(32, 3, -1))), # extra run with 0 exp
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

    if nrs_name != "MorrisUnaryHEB":

        for a1 in tqdm(fst_pos_args, desc='First argument|exp|g|size'):

            if wandb_flag:
                run = wandb.init(project='Uniform_NRS', name=run_name + "exp|g=" + str(a1), reinit=True)

            for a2 in tqdm(snd_pos_args, desc='Second argument|frac|size'):

                if nrs_name in ["Morris", "MorrisHEB", "MorrisBiasHEB"] and a1 >= a2 - 3:
                    break

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
    
    else:
        if wandb_flag:
            run = wandb.init(project='Uniform_NRS', name=run_name, reinit=True)

        for a1 in tqdm(snd_pos_args, desc='Argument|frac|size'):
            model = copy.deepcopy(og_model)
            model = trf_uniform_precision(model, nrs_name, a1, None)
            acc = eval_model(model, test_dl, device)
            if wandb_flag:
                wandb.log({'acc': acc, 'arg_1': a1})
                # save results
                results.append((a1, acc))
            else:
                print(f'Accuracy: nrs: {nrs_name}, {acc}, arg_1: {a1}')

        if wandb_flag:
            run.finish()
    
    if nrs_name != "MorrisUnaryHEB":
        # save to dataframe
        df = pd.DataFrame(results, columns=['arg_1', 'arg_2', 'acc'])
        df.to_csv(f'./results/{run_name}results.csv', index=False)
    else:
        # save to dataframe
        df = pd.DataFrame(results, columns=['arg_1', 'acc'])
        df.to_csv(f'./results/{run_name}results.csv', index=False)


def best_mixed_precision(og_model, dataset_name, nrs_name, wandb_flag, run_name=None, starting_arg1=0, starting_arg2=0):

    layer_list = list(og_model.named_parameters())

    layer_group_list = []
    for i in range(0, len(layer_list)):
        if layer_list[i][0].endswith('weight') and layer_list[i + 1][0].endswith('bias'):
            layer_group_list.append([layer_list[i], layer_list[i+1]])
        else:
            if layer_list[i][0].endswith('bias'):
                continue
            layer_group_list.append([layer_list[i]])


    layer_conf_list = [(layer_group, {"nrs": nrs_name, "a1": starting_arg1, "a2": starting_arg2}) for layer_group in layer_group_list]


    test_dl = get_test_dl(dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    uniform_model = copy.deepcopy(og_model)
    uniform_model = trf_uniform_precision(uniform_model, nrs_name, starting_arg1, starting_arg2)
    uniform_acc = eval_model(uniform_model, test_dl, device)

    print("Uniform model accuracy: ", uniform_acc)

    starting_arg1 -= 1
    starting_arg2 -= 1

    nrs_range_dict = {
        "IEEE754": (list(range(starting_arg1, 1, -1)), list(range(starting_arg2, 1, -1))),
        "Morris": (list(range(starting_arg1, 1, -1)), list(range(starting_arg2, 3, -1))),
        "MorrisHEB": (list(range(starting_arg1, 1, -1)), list(range(starting_arg2, 3, -1))),
        "MorrisBiasHEB": (list(range(starting_arg1, 1, -1)), list(range(starting_arg2, 3, -1))),
        "MorrisUnaryHEB": (None, list(range(starting_arg1, 3, -1))),
        "Posit": (list(range(starting_arg1, -1, -1)), list(range(starting_arg2, 3, -1))), # extra run with 0 exp
    }

    fst_pos_args = nrs_range_dict[nrs_name][0]  #Unary ; rest -> [0]
    snd_pos_args = nrs_range_dict[nrs_name][1]

    optimal_group_conf = []

    prev_model = copy.deepcopy(uniform_model)

    for layer_group, conf in tqdm(layer_conf_list):

        for l in layer_group:
            print()
            print(l[0])
        
        best_a1 = None
        # find best first argument
        for a1 in fst_pos_args:

            print("Current first argument: ", a1)
            
            sd = prev_model.state_dict()
            test_model = copy.deepcopy(uniform_model)

            for layer in layer_group:
                convert_layer_weigths(layer[0], layer[1], nrs_name, a1, conf["a2"])
                mod_layer = np.load(f'./weight_prc/{layer[0]}.npy').astype("<f")
                sd[layer[0]] = torch.tensor(mod_layer)

            test_model.load_state_dict(sd)
            acc = eval_model(test_model, test_dl, device)

            print("Accuracy: ", acc)

            if uniform_acc - acc < 0.01:
                prev_model = copy.deepcopy(test_model)
                best_a1 = a1
            else:
                if best_a1 is None:
                    best_a1 = conf["a1"]
                break

        if best_a1 is None:
            best_a1 = conf["a1"]

        best_a2 = None
        # find best second argument

        # morris 
        for a2 in snd_pos_args:

            if best_a1 >= a2 - 3:
                break

            print("Current second argument: ", a2)

            sd = prev_model.state_dict()
            test_model = copy.deepcopy(uniform_model)

            for layer in layer_group:
                convert_layer_weigths(layer[0], layer[1], nrs_name, best_a1, a2)
                mod_layer = np.load(f'./weight_prc/{layer[0]}.npy').astype("<f")
                sd[layer[0]] = torch.tensor(mod_layer)

            test_model.load_state_dict(sd)
            acc = eval_model(test_model, test_dl, device)

            print("Accuracy: ", acc)

            if uniform_acc - acc < 0.01:
                prev_model = copy.deepcopy(test_model)
                best_a2 = a2
            else:
                if best_a2 is None:
                    best_a2 = conf["a2"]
                break
        
        optimal_group_conf.append((layer_group, {"nrs": nrs_name, "a1": best_a1, "a2": best_a2}))
                
    for group, conf in optimal_group_conf:
        print([g[0] for g in group], conf)



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
    parser.add_argument('--start_a1', type=int, default=4, help='starting exponent value')
    parser.add_argument('--start_a2', type=int, default=32, help='starting fraction value')
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

    # python transform_eval.py --arch ResNet18 --trained_model CIFAR100_ResNet18.pth --dataset CIFAR100 --nrs_name IEEE754 --mixed --start_a1 4 --start_a2 6
    if args.mixed:
        best_mixed_precision(model, args.dataset, args.nrs_name,
                            args.wandb, f"{args.nrs_name}_",
                            starting_arg1=args.start_a1, starting_arg2=args.start_a2)

if __name__ == "__main__":
    main()