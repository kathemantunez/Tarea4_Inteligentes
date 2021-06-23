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


    print("Model structure: ", model, "\n\n")
    c = 0
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    cont=0
    """for k, v in model.state_dict().items():

        param = torch.nn.Parameter(v).data
        temp = param[:2]
       # print(temp)
        if cont == 0:
            lista.append(temp[0,0])
            lista.append(temp[0,1])
            lista.append(temp[1,0])
            lista.append(temp[1, 1])

            print(cont)

        elif cont == 1:
            lista.append(temp[0, 0])
            lista.append(temp[0, 1])
            print(cont)
        elif cont == 2:
            lista.append(temp[0, 0])
            lista.append(temp[0, 1])
            lista.append(temp[1, 0])
            lista.append(temp[1, 1])
            print(cont)
        elif cont == 3:
            lista.append(temp[0, 0])
            lista.append(temp[0, 1])

            print(cont)

        cont=cont+1
        print(cont)

"""

    guardarpkl(lista)
    lista_pesos = loadpkl()
    # imprimir valores iniciales random
""""
    print("---layer 0----")
    print("Neurona 0 --> Peso 1: ", lista_pesos[0], " Peso 2: ", lista_pesos[1], " Bias: ", lista_pesos[4])
    print("Neurona 1 --> Peso 1: ", lista_pesos[2], " Peso 2: ", lista_pesos[3], " Bias: ", lista_pesos[5])
    print("---layer 1----")
    print("Neurona 0 --> Peso 1: ", lista_pesos[6], " Peso 2: ", lista_pesos[7], " Bias: ", lista_pesos[10])
    print("Neurona 1 --> Peso 1: ", lista_pesos[8], " Peso 2: ", lista_pesos[9], " Bias: ", lista_pesos[11])
"""
    torch.save(model.state)
    print("\n FEED FORWARD")
    y = model.forward(x)


if __name__ == '__main__':
    main()
"""

#entradas tensor
x=torch.tensor([[0,0],[0,1],[1,0],[1,1]]).cuda()
#generar pesos aleatorios
pesos=torch.()

model = NeuralNetwork().to("cuda")
#model.apply(aplicar_pesos)
print("Model structure: ", model, "\n\n")
pesos=None

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print(pesos)


model = NeuralNetwork()
print(model)

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

"""
