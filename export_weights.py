
import torch

from ResNet import ResNet18


def export_as_txt():
    model = ResNet18()
    model.load_state_dict(torch.load('model.pth'))

    with open('model.txt', 'w') as f:
        for name, param in model.named_parameters():
            f.write(f'{name}\n{param.data.tolist()}\n')


if __name__ == "__main__":

    # # load model
    # model = ResNet18()
    # model.load_state_dict(torch.load("model.pth"))

    export_as_txt()

    # # print model
    # # print(model)

    # for layer in model.named_children():

    #     if list(layer[1].named_children()) != []:

    #         for sublayer in layer[1].named_children():

    #             if list(sublayer[1].named_children()) != []:

    #                 for subsublayer in sublayer[1].named_parameters():

    #                     print(layer[0], sublayer[0], subsublayer[0])

    #             else:
    #                 print(layer[0], sublayer[0])

    #     else:
    #         print(layer[0])


        
