import numpy as np
import matplotlib.pyplot as plt
from numba import jit_module

class BA:
    def __init__(self, num_pop, min_freq, max_freq, task):
        assert max_freq > 0, "Max frequency must be greater than 0"
        assert min_freq > 0, "Min frequency must be greater than 0"
        assert min_freq < max_freq, "Min frequency must be less than max frequency"

        x_min, x_max = task.x_min, task.x_max

        self.Objective = task

        self.population = np.random.rand(num_pop, task.dim) * (x_max - x_min) + x_min

        self.velocity = np.zeros((num_pop, task.dim))
        self.frequencies = (np.ones(num_pop) * min_freq)

        self.bounds = (x_min, x_max)
        self.freq = (min_freq, max_freq)

    def init_pop(self, num_pop, task, x_min, x_max):
        self.population = np.random.rand(num_pop, task.dim) * (x_max - x_min) + x_min

    def main_loop(self, max_iter, init_r, init_A, alpha, gamma, delta, times=1):
        assert 0 <= init_r <= 1, "Initial random factor must be in [0, 1]"
        assert 1 <= init_A <= 2, "Initial acceleration factor must be in [1, 2]"
        assert 0 < alpha < 1, "Alpha must be in (0, 1)"
        assert 0 < gamma < 1, "Gamma must be in (0, 1)"
        assert 0 < delta < 1, "Delta must be in (0, 1)"

        best_result = np.inf
        best_res_pos = np.zeros(self.Objective.dim)

        for _ in range(times):
            results = []

            self.init_pop(self.population.shape[0], self.Objective, self.bounds[0], self.bounds[1])

            fitness = np.apply_along_axis(self.Objective.Query, 1, self.population)
            fitness = np.where(np.isnan(fitness), np.inf, fitness) # in case of nan

            best_ind = np.argmin(fitness)
            best_pos = np.copy(self.population[best_ind])
            best_fitness = np.copy(fitness[best_ind])
            A = init_A * np.ones(self.population.shape[0])
            r = init_r * np.ones(self.population.shape[0])

            for i in range(max_iter):
                # Might be parallelized since computations for each bat is independent
                for bat in range(self.population.shape[0]):
                    # Updating frequencies, velocities and positions
                    self.frequencies[bat] = np.random.rand() * (self.freq[1] - self.freq[0]) + self.freq[0]
                    self.velocity[bat] += (- self.population[bat] + best_pos) * self.frequencies[bat]
                    self.population[bat] += self.velocity[bat]

                    x_new = np.copy(self.population[bat])

                    if np.random.rand() > r[bat]:
                        x_new = best_pos + (1 - 2 * np.random.rand()) * A.mean()

                    np.clip(self.population[bat], *self.bounds, self.population[bat])
                    np.clip(x_new, *self.bounds, x_new)

                    # np.random.rand() < A[bat]
                    fit_new = self.Objective.Query(x_new)
                    if (not np.isnan(fit_new)) and fit_new < self.Objective.Query(self.population[bat]):
                        self.population[bat] = np.copy(x_new)
                        fitness[bat] = np.copy(self.Objective.Query(x_new))

                        r[bat] = init_r * (1 - np.exp(-gamma * i))
                        A[bat] = alpha * A[bat]

                fitness = np.apply_along_axis(self.Objective.Query, 1, self.population)
                fitness = np.where(np.isnan(fitness), np.inf, fitness) # in case of nan
                min_fit = np.copy(np.min(fitness))

                if min_fit < best_fitness or np.isnan(min_fit):
                    best_ind = np.argmin(fitness)
                    best_pos = np.copy(self.population[best_ind])
                    best_fitness = np.copy(fitness[best_ind])

                results.append(best_fitness)

                if best_result > best_fitness:
                    best_result = np.copy(best_fitness)
                    best_res_pos = np.copy(best_pos)

                print("Iteration: %d\nMin value: %.3e\nMin position: %s" % (i + 1, best_fitness, best_pos))

            plt.plot(results)

        plt.hlines(xmin=0, xmax=max_iter, y=best_result, linestyles='--')
        plt.show()

        return best_result, best_res_pos
    
class CO:
    """
    Implementation of the cuckoo search algorithm.
    """

    def __init__(self, num_pop, p_detect, delta, task):
        """
        Constructor for Classical Optimization algorithm.
        
        Arguments:
            num_pop: Number of population members.
            num_dim: Number of dimensions in the problem.
            p_detect: Probability of detecting an egg.
            delta: Step size for generating new positions.
            x_min: Minimum value for each dimension in the problem.
            x_max: Maximum value for each dimension in the problem.
        """
        # Checking delta and p_detect
        assert 0 <= p_detect <= 1,  "Delta must be in [0, 1]"
        assert 0 < delta < 1, "Delta must be in (0, 1)"

        self.task = task
        num_dim = task.dim
        x_min, x_max = task.x_min, task.x_max

        # Inital population
        self.population = np.random.rand(num_pop, num_dim) * (x_max - x_min) + x_min
        self.bounds = (x_min, x_max)

        # Probability of detecting an egg during the search.
        self.p_detect = p_detect

        #Step size for generating new positions.
        self.delta = delta

    def init_pop(self, num_pop, num_dim, x_min, x_max):
        self.population = np.random.rand(num_pop, num_dim) * (x_max - x_min) + x_min
        

    def main_loop(self, max_iter, cookoo=1, times=1):
        """
        Cuckoo search algorithm main loop.

        Arguments:
            max_iter: Maximum number of iterations.
            objective_function: Function to optimize, must accept a position as input and return a float.

        Returns:
            The best fitness and the corresponding best position.
        """

        best_res = np.inf
        best_res_pos = np.zeros(self.population.shape[1])

        for time in range(times):
            results = []

            self.init_pop(self.population.shape[0], self.population.shape[1], self.bounds[0], self.bounds[1])

            best_pos = np.random.rand(self.population.shape[1])
            best_fitness = self.task.Query(best_pos)

            if np.isnan(best_fitness):
                best_fitness = np.inf

            fitness = np.apply_along_axis(self.task.Query, 1, self.population)
            fitness = np.where(np.isnan(fitness), np.inf, fitness)

            for i in range(max_iter):
                for _ in range(cookoo):
                    # Choosing nest
                    k = np.random.randint(0, self.population.shape[0])

                    # Generating new position
                    x_cur = self.population[k] + self.delta * (self.bounds[1] - self.bounds[1]) * (2 * np.random.rand() - 1)
                    np.clip(x_cur, *self.bounds)

                    # Updating best position
                    cur_fit = self.task.Query(x_cur)

                    if np.isnan(cur_fit):
                        cur_fit = np.inf

                    if cur_fit < best_fitness:
                        best_fitness = np.copy(self.task.Query(x_cur))
                        best_pos = np.copy(x_cur)
                        self.population[k] = np.copy(x_cur)

                    # Detecting egg
                    if np.random.rand() < self.p_detect:
                        m = np.argmax(fitness)
                        
                        # Updating worst nest
                        self.population[m] += self.delta * (self.bounds[1] - self.bounds[1]) * (2 * np.random.rand() - 1)
                        np.clip(self.population[m], *self.bounds)

                results.append(best_fitness)

                if best_res > best_fitness:
                    best_res = np.copy(best_fitness)
                    best_res_pos = np.copy(best_pos)

                print("Iteration: %d\nMin value: %.3e\nMin position: %s" % (i + 1, best_fitness, best_pos))

            plt.plot(range(max_iter), results)

        plt.hlines(xmin=0, xmax=max_iter, y=best_res, linestyles='--')
        plt.show()
        return best_res, best_res_pos


jit_module(nopython=True, error_model="numpy")