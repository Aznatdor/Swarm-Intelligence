import numpy as np

def get_data(filepath):
    weights, costs = [], []
    with open(filepath) as file:
        weight = int(file.readline())

        for line in file.readlines():
            w, c = map(int, line.split())
            weights.append(w)
            costs.append(c)

    return weight, np.array(weights), np.array(costs)


def create_test_case(num, weight):
    current_weight = weight
    weights = []
    while current_weight > 1 and len(weights) < num:
        to_add = np.random.randint(1, current_weight)
        weights.append(to_add)
        current_weight -= to_add

    left = num - len(weights)
    if left:
        weights.extend(np.random.randint(1, weight, size=(left,)))

    costs = np.random.randint(0, weight, size=(num,))

    return weights, costs