import numpy as np

def euclidean_distance(x, y):
    if len(x) != len(y):
        return None
    squared_distance = 0
    for i in range(len(x)):
        squared_distance += (x[i] - y[i]) ** 2
    return squared_distance ** 0.5