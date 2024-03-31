import numpy as np

def create_circle(n, radius=1):
    circle = []
    theta = 0
    delta = 2 * np.pi / n

    side_length = 2 * radius * np.sin(delta / 2)

    for _ in range(n):
        x = np.cos(theta)
        y = np.sin(theta)
        circle.append(np.array([x, y]))
        theta += delta


    return np.array(circle), n * side_length

def read_tsp_file(filepath):
    tsp = []
    with open(filepath) as file:

        for line in file.readlines():
            if line[0].isdigit():
                _, x, y = line.split()
                tsp.append(np.array([np.float32(x), np.float32(y)]))


    return np.array(tsp)