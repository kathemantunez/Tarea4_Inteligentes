import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle


#pesos
def guardarpkl(pesos):
    ficheroname='pesos_parte_1'
    outfile=open(ficheroname,'wb')
    pickle.dump(pesos,outfile)
    outfile.close()

def loadpkl():
    fichero=open('pesos_parte_1','rb')
    lista_fichero=pickle.load(fichero)
    fichero.close()
    return lista_fichero

def aplicar_pesos(m):
    fichero=loadpkl()
    print(fichero)

    if type(m)==nn.Linear:
        torch.nn.init.eye_(fichero)
        m.bias.data.fill_(0.01)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#entradas tensor
x=torch.tensor([[0,0],[0,1],[1,0],[1,1]]).cuda()
#generar pesos aleatorios
pesos=torch.rand()

model = NeuralNetwork().to("cuda")
#model.apply(aplicar_pesos)
print("Model structure: ", model, "\n\n")
pesos=None

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print(pesos)

"""
model = NeuralNetwork()
print(model)

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

"""
