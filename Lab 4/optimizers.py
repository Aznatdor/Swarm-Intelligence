import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

class ACO:
    def __init__(self, graph, *, init_pheromone=1):
        n = len(graph)

        self.graph = graph

        self.population = [[i] for i in range(n)] * n # Each ant starts at city number 0
        self.dist = np.zeros(shape=(n)) # dist[ant] = distance made so far

        self.pheromone = np.ones((n, n)) * init_pheromone

        self.adjacency_matrix = np.zeros(shape=(n, n), dtype=np.float32) # [i, j] = Euclidean distance between city i and j

        # Setting up the distances
        for i in range(n):
            self.adjacency_matrix[i] = np.apply_along_axis(np.linalg.norm, 1, graph[i] - graph)

        # Banning paths from city to itself 
        self.adjacency_matrix = np.where(self.adjacency_matrix == 0, np.inf, self.adjacency_matrix)
        
    
    def main_loop(self, num_iter, alpha, beta, Q, rho):
        n = self.adjacency_matrix.shape[0]
        best_route = None
        min_dist = np.inf
        
        mu = 1 / self.adjacency_matrix
        ants = np.array(range(len(self.population)))

        for i in range(num_iter):
            self.population = [[0]] * len(self.population) # Each ant starts at city number 0
            self.dist = np.zeros(self.dist.shape)
            delta_tau = np.zeros_like(self.pheromone) # To perform pheromon decay after loop ends
            
            self.fast_ant(self, ants, alpha=alpha, mu=mu, beta=beta, n=n, Q=Q, delta_tau=delta_tau)

            min_ant = np.argmin(self.dist[self.dist != 0])

            if min_dist > self.dist[min_ant] and self.dist[min_ant] != 0:
                min_dist = np.copy(self.dist[min_ant])
                best_route = np.copy(self.population[min_ant])

            print("Iteration: %i\nMin distance: %.3f\nMin route %s" % (i + 1, min_dist, best_route)) 

            # Pheromone decay
            self.pheromone = (1 - rho) * self.pheromone + delta_tau
        
        return min_dist, best_route
    

    @np.vectorize(excluded=['self','alpha', 'mu', 'beta', 'n', 'Q', 'delta_tau'])
    def fast_ant(self, ant, alpha, mu, beta, n, Q, delta_tau):
        while len(self.population[ant]) != n:
            # Getting the last city that ant have visited so far
            current_city = self.population[ant][-1]
            # Temporary variable to compute probabilities later
            tmp = (self.pheromone[current_city] ** alpha) * (mu[current_city] ** beta)

            # Setting visited cities probabilities to zero
            for j in range(n):
                if j in self.population[ant]:
                    tmp[j] = 0

            # Probabilities itself
            prob = tmp / tmp.sum()

            # Cumulative probabilites to choose next city
            cum_prob = np.cumsum(prob)

            # Choosing next city while it hasn't been visited
            next_city = np.searchsorted(cum_prob, np.random.rand())

            while next_city in self.population[ant]:
                next_city = np.searchsorted(cum_prob, np.random.rand())

            self.population[ant].append(next_city)
            self.dist[ant] += self.adjacency_matrix[current_city][next_city]

            # Updating pheromon
            self.pheromone[next_city][current_city] += Q / self.dist[ant]
            self.pheromone[current_city][next_city] += Q / self.dist[ant]

            delta_tau[next_city][next_city] += Q / self.dist[ant]
            delta_tau[current_city][next_city] += Q / self.dist[ant]

            if len(self.population[ant]) == n:
                # Connecting the last city with the first
                self.dist[ant] += self.adjacency_matrix[0][self.population[ant][-1]]
    

    def main_loop_plot(self, num_iter, alpha, beta, Q, rho, gif_name="ACO"):
        n = self.adjacency_matrix.shape[0]
        best_route = None
        min_dist = np.inf
        ants = np.array(range(len(self.population)))
        
        mu = 1 / self.adjacency_matrix

        # plotting
        fig, ax = plt.subplots()
        images = []
        canvas = FigureCanvasAgg(fig)

        for i in range(num_iter):
            self.population = [[0]] * len(self.population) # Each ant starts at city number 0
            self.dist = np.zeros(self.dist.shape)
            delta_tau = np.zeros_like(self.pheromone) # To perform pheromon decay after loop ends
            
            self.fast_ant(self, ants, alpha=alpha, mu=mu, beta=beta, n=n, Q=Q, delta_tau=delta_tau)

            min_ant = np.argmin(self.dist[self.dist != 0])

            if min_dist > self.dist[min_ant] and self.dist[min_ant] != 0:
                min_dist = np.copy(self.dist[min_ant])
                best_route = np.copy(self.population[min_ant])

                arr_to_plot = self.graph[best_route]

                plt.title("Ant Colony Algorithm")
                ax.plot(arr_to_plot[:, 0], arr_to_plot[:, 1], label="dist %.3f" % min_dist)
                ax.scatter(self.graph[:, 0], self.graph[:, 1], color="red")
                plt.legend(loc="best")

                # Render the plot as an RGBA buffer
                canvas.draw()
                buf = canvas.buffer_rgba()

                # Create a PIL Image from the RGBA buffer
                image = Image.frombytes('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA')
                images.append(image)
                plt.cla()

            print("Iteration: %i\nMin distance: %.3f\nMin route %s" % (i + 1, min_dist, best_route)) 

            # Pheromone decay
            self.pheromone = (1 - rho) * self.pheromone + delta_tau

        # something like pause    
        images.extend([images[-1]] * max(100, num_iter // 25 + 1))

        images[0].save(f'{gif_name}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=10, loop=0)
        
        return min_dist, best_route
    

class GO:
    def __init__(self, num_pop, graph):
        n = len(graph)
        self.graph = graph
        self.num_pop = num_pop
        self.population = np.empty(shape=(num_pop, len(graph)), dtype=np.int32)

        # Setting up the initial population as random permutation of cities
        for i in range(num_pop):
            self.population[i] = np.random.permutation(len(graph))

        self.adjacency_matrix = np.zeros(shape=(n, n), dtype=np.float32) # [i, j] = Euclidean distance between city i and j

        # Setting up the distances
        for i in range(n):
            self.adjacency_matrix[i] = np.apply_along_axis(np.linalg.norm, 1, graph[i] - graph)

        # Banning paths from city to itself 
        self.adjacency_matrix = np.where(self.adjacency_matrix == 0, np.inf, self.adjacency_matrix)


    def helper(self, child, parent1, parent2, cut_left, cut_right):
        """
        Helper function to create a child from two parents

        Choosing two cuts to excange gens, then adding rest of the genes in one of the parent's order
        """
        n = len(parent1)
        for i in range(cut_left, cut_right):
            if parent2[i] not in child:
                for j in range(n):
                    if child[j] == -1:
                        child[j] = parent2[i]
                        break

        for i in range(n):
            if parent1[i] not in child:
                for j in range(n):
                    if child[j] == -1:
                        child[j] = parent1[i]
                        break

    # Seems to work better
    def helper2(self, child, parent1, parent2, cut_left, cut_right):
        """
        Helper function to create a child from two parents

        Choosing two cuts to excange gens, then adding rest of the genes in one of the parent's order
        """
        n = len(parent1)
        i = (cut_right + 1) % n # Child index
        p = i # Parent index

        child[cut_left:cut_right + 1] = parent1[cut_left:cut_right + 1]

        # While there's empty places in the child
        while -1 in child:
            # If parent's city is not in child
            if parent2[p] not in child:
                if child[i] == -1:
                    child[i] = parent2[p]
                i += 1
                i %= n
            else:
                p = (p + 1) % n


    def crossover(self, p1, p2, mutation_chance=0.3):
        """
        Function to perform crossover
        """
        n = self.population.shape[1]
        cut_left, cut_right = np.random.randint(0, n), np.random.randint(0, n)

        while cut_left == cut_right:
            cut_left = np.random.randint(0, n)

        if cut_left > cut_right:
            cut_left, cut_right = cut_right, cut_left     

        child1, child2 = -np.ones((2, n), dtype=np.int32)

        self.helper2(child1, p1, p2, cut_left, cut_right)
        self.helper2(child2, p2, p1, cut_left, cut_right)

        self.mutation(child1, mutation_chance)
        self.mutation(child2, mutation_chance)

        self.population = np.append(self.population, [child1, child2], axis=0)


    def mutation(self, child, mutation_chance):
        """
        Function to perform mutation
        """
        if np.random.rand() < mutation_chance:
            i, j = np.random.randint(0, len(child)), np.random.randint(0, len(child))
            child[i], child[j] = child[j], child[i]


    def fitness(self, child):
        """
        Function to compute the fitness of a child. Total length of the path
        """
        res = 0
        for i in range(len(child) - 1):
            res += self.adjacency_matrix[child[i]][child[i + 1]]
        res += self.adjacency_matrix[child[-1]][child[0]]

        return res
    

    @np.vectorize(excluded=["mutation_chance"])
    def fast_crossover(self, _, *, mutation_chance):
        # Choose two random parents
        p1, p2 = np.random.randint(0, len(self.population)), np.random.randint(0, len(self.population))

        while p1 == p2:
            p1, p2 = np.random.randint(0, len(self.population)), np.random.randint(0, len(self.population))

        # Performing crossover
        self.crossover(self.population[p1], self.population[p2], mutation_chance=mutation_chance)


    def main_loop(self, num_iter, num_child, mutation_chance=0.3):
        fitness = np.apply_along_axis(self.fitness, 1, self.population)

        childs = np.arange(num_child)

        best_index = np.argmin(fitness[fitness != 0])

        best_res = np.copy(fitness[fitness != 0][best_index])
        best_route = np.copy(self.population[fitness != 0][best_index])

        for i in range(num_iter):
            self.fast_crossover(self, childs, mutation_chance=mutation_chance)

            # Computing fitness
            fitness = np.apply_along_axis(self.fitness, 1, self.population)

            # Selecting best
            sorted_indeces = np.argsort(fitness)[:self.num_pop]
            fitness = fitness[sorted_indeces]
            self.population = self.population[sorted_indeces]

            best_index = np.argmin(fitness[fitness != 0])

            if best_res > fitness[fitness != 0][best_index]:
                best_res = np.copy(fitness[fitness != 0][best_index])
                best_route = np.copy(self.population[fitness != 0][best_index])

            print("Iteration: %i\nMin distance: %.3f\nMin route %s" % (i + 1, best_res, best_route))

        return best_res, best_route
    

    def main_loop_plot(self, num_iter, num_child, mutation_chance=0.3, gif_name="GO"):
        fitness = np.apply_along_axis(self.fitness, 1, self.population)

        childs = np.arange(num_child)

        best_index = np.argmin(fitness[fitness != 0])

        min_dist = np.copy(fitness[fitness != 0][best_index])
        best_route = np.copy(self.population[fitness != 0][best_index])

        # plotting
        fig, ax = plt.subplots()
        images = []
        canvas = FigureCanvasAgg(fig)

        for i in range(num_iter):
            self.fast_crossover(self, childs, mutation_chance=mutation_chance)

            # Computing fitness
            fitness = np.apply_along_axis(self.fitness, 1, self.population)

            # Selecting best
            sorted_indeces = np.argsort(fitness)[:self.num_pop]
            fitness = fitness[sorted_indeces]
            self.population = self.population[sorted_indeces]

            best_index = np.argmin(fitness[fitness != 0])

            if min_dist > fitness[fitness != 0][best_index]:
                min_dist = np.copy(fitness[fitness != 0][best_index])
                best_route = np.copy(self.population[fitness != 0][best_index])

                arr_to_plot = self.graph[best_route]

                plt.title("Genetic Algorithm")
                ax.plot(arr_to_plot[:, 0], arr_to_plot[:, 1], label="dist %.3f" % min_dist)
                ax.scatter(self.graph[:, 0], self.graph[:, 1], color="red")
                plt.legend(loc="best")

                # Render the plot as an RGBA buffer
                canvas.draw()
                buf = canvas.buffer_rgba()

                # Create a PIL Image from the RGBA buffer
                image = Image.frombytes('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA')
                images.append(image)
                plt.cla()

            print("Iteration: %i\nMin distance: %.3f\nMin route %s" % (i + 1, min_dist, best_route))

        # something like pause    
        images.extend([images[-1]] * max(100, num_iter // 25 + 1))

        images[0].save(f'{gif_name}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=10, loop=0)

        return min_dist, best_route