import numpy as np

def stepfunction(soma):
    if (soma >= 1):
        return 1
    return 0

def sigmoidfunction(soma):
    return 1 / (1 + np.exp(-soma))

def tahnfunction(soma):
    return(np.exp(soma) - (np.exp(-soma))) / (np.exp(soma) + (np.exp(-soma)))

def relufunction(soma):
    if soma >= 0:
        return soma
    return 0

def linearfunction(soma):
    return soma

def softmaxfunction(x):
    ex = np.exp(x)
    return ex / ex.sum()
    

testeStep = stepfunction(1)
teste = sigmoidfunction(0.358)
teste = tahnfunction(-0.358)
teste = relufunction(0.358)
teste = linearfunction(0.358)
valores = ()
#print(softmaxfunction(valores))

