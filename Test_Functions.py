import numpy as np

def rosenbrock(x):  
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    for i in range(len(x)):
        for j in range(dimension - 1):
            sum[i] += (np.square(1 - x[i][j]) + 100 * np.square(x[i][j + 1] - np.square(x[i][j])))
    sum = np.array(sum)
    return sum
def rastrigin(x):  
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (np.square(x[i][j]) - 10 * np.cos(2 * np.pi * x[i][j]) + 10)
    sum = np.array(sum)
    return sum
def sphere(x):  
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += np.square(x[i][j])
    sum = np.array(sum)
    return sum
def ellipsoid(x):  
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]

    dimension = x.shape[1]
    sum = [0.0] * x.shape[0]
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (j + 1) * np.square(x[i][j])
    sum = np.array(sum)
    return sum
def ackley(x): 
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    for i in range(len(x)):
        atemp1 = 0.0
        atemp2 = 0.0
        for j in range(dimension):
            atemp1 += np.square(x[i][j])
            atemp2 += np.cos(2 * np.pi * x[i][j])
        sum[i] = -20 * np.exp(-0.2 * np.sqrt(atemp1 / dimension)) - np.exp(atemp2 / dimension) + 20 + np.exp(1)
    sum = np.array(sum)
    return sum
def griewank(x):  
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    for i in range(len(x)):
        atemp1 = 0.0
        atemp2 = 1
        for j in range(dimension):
            atemp1 += np.square(x[i][j])
            atemp2 *= np.cos(x[i][j] / np.sqrt(j + 1))
        sum[i] = (atemp1 / 4000) - atemp2 + 1
    sum = np.array(sum)
    return sum