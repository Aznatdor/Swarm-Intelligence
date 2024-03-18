import numpy as np
from numba import float32, int64, jit
from numba.experimental import jitclass

spec = [
    ("y_a", float32),
    ("y_b", float32),
    ("num_ticks", int64),
    ("dim", int64),
    ("h", float32),
    ("t_init", float32),
    ("x_min", float32),
    ("x_max", float32),
]

@jitclass(spec)
class Boundary1:
    def __init__(self, y_a, y_b, num_ticks, a, b, alpha_min, alpha_max):
        self.y_a = y_a
        self.y_b = y_b
        self.num_ticks = num_ticks
        self.h = (b - a) / num_ticks
        self.t_init = a
        self.x_min = alpha_min
        self.x_max = alpha_max
        self.dim = 1

    def Query(self, alpha):
        t = self.t_init
        y = np.zeros(self.num_ticks + 1, dtype=np.float32)
        z = np.zeros(self.num_ticks + 1, dtype=np.float32)

        func = (lambda t : np.exp(- t) * np.sin(t))

        y[0] = self.y_a
        z[0] = alpha[0]
        for i in range(1, self.num_ticks + 1):
            y[i] = y[i - 1] + self.h * z[i - 1]
            z[i] = z[i - 1] + self.h * (func(t) - (y[i - 1] + z[i - 1])) 
            t += self.h

        return (y[-1] - self.y_b) ** 2

spec_MB = [
    ("x_min", float32[:]),
    ("x_max", float32[:]),
    ("dim", int64),
    ("min_point", float32[:]),
    ("min", float32)
]

@jitclass(spec_MB)
class M_B:
    def __init__(self):
        self.x_min = np.array([-10, -6.5], dtype=np.float32)
        self.x_max = np.array([0, 0], dtype=np.float32)
        self.dim = 2

        self.min = -106.7645367
        self.min_point = np.array([-3.1302468, -1.5821422], dtype=np.float32) 

    def Query(self, vec):
        x, y, = vec[0], vec[1]

        if (x + 5) ** 2 + (y + 5) ** 2 > 25:
            return np.inf
        else:
            return np.exp((1 - np.cos(x)) ** 2) * np.sin(y) + np.exp((1 - np.sin(y)) ** 2) * np.cos(x) + (x - y) ** 2
        
spec_ras = [
    ("dim", int64),
    ("x_min", float32),
    ("x_max", float32),
    ("A", float32),
    ("min_point", float32[:]),
    ("min", float32)
]

@jitclass(spec_ras)
class Rastrigin:
    def __init__(self, dim, A):
        self.dim = dim
        self.x_min = -5.12
        self.x_max = 5.12
        self.A = A

        self.min_point = np.zeros(dim, dtype=np.float32)
        self.min = 0

    def Query(self, vec):
        return self.A * vec.shape[0] + (vec ** 2 - self.A * np.cos(2 * np.pi * vec)).sum() 


spec_A1 = [
    ("num_ticks", int64),
    ("x_min", float32),
    ("x_max", float32),
    ("dim", int64),
    ("h", float32),
    ("y_a", float32),
    ("x_b", float32),
    ("x_prime_a", float32),
    ("query_type", int64)
]

'''
Min Point: [-1.59943418]	Min Value: 9.26e-06
'''

@jit("float32(float32)")
def type_1_error_single(x):
    return abs(x - 2)

@jit("float32(float32)")
def type_2_error_single(x):
    return (x - 2) ** 2

@jitclass(spec_A1)
class Assignment1:
    def __init__(self, num_ticks, query_type):
        # Setting parameters
        self.num_ticks = num_ticks
        self.x_min = -3 # min alpha
        self.x_max = 3 # max alpha
        self.dim = 1
        self.h = (3 - 1) / num_ticks
        self.query_type = query_type

        # Setting constraints
        self.y_a = 1
        self.x_b = 2
        self.x_prime_a = 1

    def Query(self, x_a):
        global type_1_error_single, type_2_error_single
        error_functions = [
            type_1_error_single,
            type_2_error_single
        ]
        t = 1

        x = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        y = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        z = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)

        z[0] = self.x_prime_a
        y[0] = self.y_a
        x[0] = x_a[0]

        for i in range(1, self.num_ticks + 1):
            z[i] = z[i - 1] + self.h * ( 
                (x[i - 1] * t) ** 2 - ((y[i - 1] ** 4 + x[i - 1] ** 3 - 3 * np.sin(t * z[i - 1])) ** 2) * np.cos(z[i - 1]))
            y[i] = y[i - 1] + self.h * (y[i - 1] ** 4 + x[i - 1] ** 3 - 3 * np.sin(t * z[i - 1]))
            x[i] = x[i - 1] + self.h * z[i - 1]
            t += self.h

        return error_functions[self.query_type](x[-1])


spec_A4 = [
    ("num_ticks", int64),
    ("x_min", float32),
    ("x_max", float32),
    ("dim", int64),
    ("h", float32),
    ("x_a", float32),
    ("x_b", float32),
    ("y_a", float32),
    ("y_b", float32),
    ("query_type", int64)
]

'''
Min Point: [4.30142494 1.22941376]	Min Value: 9.69e-07
'''
@jitclass(spec_A4)
class Assignment4:
    def __init__(self, query_type):
        # Setting parameters
        self.num_ticks = 1000
        self.x_min = -8 # min alpha
        self.x_max = 8 # max alpha
        self.dim = 2
        self.h = 2 / self.num_ticks
        self.query_type = query_type

        # Setting constraints
        self.x_a = 1
        self.y_a = -1
        self.x_b = 10
        self.y_b = 21

    def Query(self, vec):
        z_a, w_a = vec # x_prime_a, y_prime_a

        t = 1

        x = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        y = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        z = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        w = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)

        x[0] = self.x_a
        y[0] = self.y_a
        z[0] = z_a
        w[0] = w_a

        for i in range(1, self.num_ticks + 1):
            x[i] = x[i - 1] + self.h * z[i - 1]
            y[i] = y[i - 1] + self.h * w[i - 1]
            z[i] = z[i - 1] + self.h * (np.cos(x[i - 1] * y[i - 1] - 
                                               np.sin(y[i - 1] + t * x[i - 1])/((x[i - 1] ** 2 + w[i - 1]) ** 2 + 1)))

            w[i] = w[i - 1] + self.h * (6 * t - (1 / t ** 4) + 
                        np.cos(5) - 4 + w[i - 1] / (t ** 2) - np.cos(3 * x[i - 1] - w[i - 1]) + x[i - 1] ** 2 / (t ** 4))
            
            t += self.h

        return (x[-1] - self.x_b) ** 2 + (y[-1] - self.y_b) ** 2
    

'''
Min Point: [ 1.51187253 -0.62497578]	Min Value: 0.00000
Min Point: [ 1.51211884 -0.62501576]	Min Value: 5.36e-07
'''

@jit("float32(float32, float32)")
def type_1_error(x, y):
    return abs(x - 5) + abs(y + 1)

@jit("float32(float32, float32)")
def type_2_error(x, y):
    return (x - 5) ** 2 + (y + 1) ** 2

@jit("float32(float32, float32)")
def type_3_error(x, y):
    return max(abs(x - 5), abs(y + 1))

@jitclass(spec_A4)
class Assignment5:
    def __init__(self, query_type):
        # Setting parameters
        self.num_ticks = 1000
        self.x_min = -30 # min alpha
        self.x_max = 30 # max alpha
        self.dim = 2
        self.h = 2 / self.num_ticks
        self.query_type = query_type

        # Setting constraints
        self.x_a = 2
        self.y_a = -1
        self.x_b = 5
        self.y_b = -1

    def Query(self, vec):
        global type_1_error, type_2_error, type_3_error
        error_functions = [
            type_1_error,
            type_2_error,
            type_3_error
        ]
        z_a, w_a = vec # x_prime_a, y_prime_a

        t = 1

        x = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        y = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        z = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)
        w = np.empty(shape=(self.num_ticks + 1), dtype=np.float32)

        x[0] = self.x_a
        y[0] = self.y_a
        z[0] = z_a
        w[0] = w_a

        for i in range(1, self.num_ticks + 1):
            x[i] = x[i - 1] + self.h * z[i - 1]
            y[i] = y[i - 1] + self.h * w[i - 1]
            z[i] = z[i - 1] + self.h * (
                np.exp(- t ** 2) - 4 * np.exp(-np.abs(x[i - 1] * w[i - 1]) * np.cos(np.sin((5 * z[i - 1] ** 2) * y[i - 1] + t ** 2)) -
                np.log(3 * t ** 2 + (t * x[i - 1] + w[i - 1]) ** 2))
            )
            w[i] = w[i - 1] + self.h * (
                np.exp(-t) * np.cos(2 * t) + np.cos(np.abs((3 * x[i - 1] - w[i - 1]) / (z[i - 1] ** 2 + 1)) -
                np.log(10 + np.exp(-np.abs( (x[i - 1] ** 2) * (w[i - 1] ** 3) ))))
            ) 

            t += self.h

        res = error_functions[int(self.query_type)](x[-1], y[-1])

        return res