
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import struct

from tqdm import tqdm
from ResNet import ResNet18


def export_as_txt(nrs_name: str, nrs_a1: str, nrs_a2: str):
    
    model = ResNet18(num_classes=100)
    model.load_state_dict(torch.load('CIFAR100_ResNet18.pth'))

    sd = model.state_dict()

    model.eval()

    with torch.no_grad():

        mod_layer = None
        mod_layer_name = None

        for name, param in tqdm(model.named_parameters(), total=len(list(model.named_parameters())), desc='Computing weights'):
            with open(f'./weight_trf/{name}.txt', 'w') as f:
                f.write(f'{name}\n{param.data.tolist()}\n')
                f.write(str(param.data.tolist()))

            mod_layer_name = name

            # transform the layer weight numerical representation system
            # os.system(f'java -jar ConvertWeights-assembly-0.1.jar ./weight_trf/{name}.txt ./weight_prc/{name}.txt {name} {nrs_name} {nrs_a1} {nrs_a2}')

            subprocess.check_call(['java', '-Xmx1024m', '-jar', 'ConvertWeights-assembly-0.1.jar', f'./weight_trf/{name}.txt', f'./weight_prc/{name}.txt', name, nrs_name, nrs_a1, nrs_a2],
                                )

            with open(f'./weight_prc/{name}.txt', 'r') as f:
                # read with format
                name = f.readline().strip()
                param = f.readline().strip()

                # convert to tensor
                param = torch.tensor(eval(param), dtype=torch.float32)   

                # modify state dict
                sd[name] = param


        torch.save(sd, 'model_prc.pth')


    # java -jar ConvertWeights-assembly-0.1.jar model.txt out_model.txt layer4.0.conv2.weight IEEE754 8 5

if __name__ == "__main__":

    # nrs = {
    #         'IEEE754':,
    #         'Morris': ,
    #         'MorrisHEB': ,
    #         'MorrisBiasHEB',: 
    #         'Posit': ,
    #         'MorrisBiasHEB':  
    #     }

    export_as_txt('IEEE754', '8', '23')

    # modify_model_weights()


        
