import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import optim

import pickle

from NeuralNetwork import NeuralNetwork


def guardarpkl(lista):
    ficheroname = 'pesos_parte_1'
    outfile = open(ficheroname, 'wb')
    pickle.dump(lista, outfile)
    outfile.close()


def loadpkl():
    fichero = open('pesos_parte_1', 'rb')
    lista_fichero = pickle.load(fichero)
    fichero.close()
    return lista_fichero


def aplicar_pesos(m):
    fichero = loadpkl()
    print(fichero)

    if type(m) == nn.Linear:
        torch.nn.init.eye_(fichero)
        m.bias.data.fill_(0.01)


def feed_forward(lista, x):
    x = 0


def main():
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    # generar pesos aleatorios
    # pesos = torch.rand()
    lista = []
    model = NeuralNetwork()




    print("MODELO: ", model, "\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print("PESOS")
    for k, v in model.state_dict().items():
        param = torch.nn.Parameter(v).data
        temp = param[:2]
        print(temp)

    # Print optimizer's state_dict
    print("OPTIMIZACION")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    #s=NeuralNetwork.forward(model,x)
    #print(s)




if __name__ == '__main__':
    main()


