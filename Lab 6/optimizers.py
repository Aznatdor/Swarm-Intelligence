"""
Module with optimization algrothms
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

class PSO:
    def __init__(self, num_pop, num_dim, cognitive, social, bounds, min_velocity, max_velocity):
        '''
        num_pop: number of particles
        num_dim: dimesionality
        cognitive: cognitive component coeficient
        social: social component coeficient
        left_bound: left bound of particles search area
        right_bound right bound of particles search area
        min_velocity: minimal allowed velocity
        max_velocity: maximum allowed velocity
        '''
        assert 0 < cognitive < 4, "Cognitive coeficient must be in (0, 4)"
        assert 0 < social < 4, "Social coeficient must be in (0, 4)"
        assert np.all(max_velocity > 0), "Maximum velocity must be positive"

        self.num_pop = num_pop
        self.num_dim = num_dim

        self.cognitive = cognitive
        self.social = social

        self.left_bound, self.right_bound = bounds[:, 0], bounds[:, 1]

        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        # Initializing particles
        
        self.positions = np.random.rand(self.num_pop, self.num_dim) * (self.right_bound - self.left_bound) + self.left_bound
        
        self.best_positions = np.copy(self.positions) # initializing best positions as starting positions

        self.velocities = np.random.rand(self.num_pop, self.num_dim) * (self.max_velocity - self.min_velocity) + self.min_velocity

    def clip_position_higer(self, position):
        return np.where(position > self.right_bound, self.right_bound - abs(position - self.right_bound), position)
    
    def clip_positon_low(self, position):
        return np.where(position > self.left_bound, self.left_bound + abs(position - self.left_bound), position)
    
    def plot_regression(self, model, params, ax):
        left, right = min(model.X_train), max(model.X_train)
        r = np.linspace(left, right, 100)
        ax[0].scatter(model.X_train, model.y_train, label="Real data")
        ax[0].scatter(model.X_train, model.fit_function(model.X_train, params), label="Train")
        ax[0].plot(r, model.fit_function(r, params))
        ax[0].set_title("Training set")
        ax[0].legend()

        left, right = min(model.X_test), max(model.X_test)
        r = np.linspace(left, right, 100)
        ax[1].scatter(model.X_test, model.y_test, label="Real data")
        ax[1].scatter(model.X_test, model.fit_function(model.X_test, params), label="Test")
        ax[1].plot(r, model.fit_function(r, params))
        ax[1].set_title("Test set")
        ax[1].legend()
    
    def main_loop(self, model, num_iter=20, *, verbose=True, gif_name="PSO"):
        train_conv, test_conv = [], []

        # plotting
        n = num_iter // 5
        fig_convegence, ax_convergence = plt.subplots()
        fig_regression, ax_regression = plt.subplots(1, 2, figsize=(10, 5))
        images_convergence = []
        images_regression = []
        canvas_convergence = FigureCanvasAgg(fig_convegence)
        canvas_regression = FigureCanvasAgg(fig_regression)

        objective_function = model.loss
        best_fittnes = np.apply_along_axis(objective_function, axis=1,  arr=self.positions)

        leader = np.argmin(best_fittnes)

        # Initializing result variables
        global_minimum_position = self.positions[leader]
        global_minimum_value = np.min(best_fittnes)

        for i in range(1, num_iter + 1):
            # Random vectors
            r1 = np.random.rand(1, self.num_dim)
            r2 = np.random.rand(1, self.num_dim)

            # Modifying velocities
            self.velocities += (self.cognitive * (self.best_positions - self.positions) * r1 + 
                                    self.social * (global_minimum_position - self.positions) * r2)
            
            self.velocities = np.maximum(self.velocities, self.min_velocity)
            self.velocities = np.minimum(self.velocities, self.max_velocity)
            
            # Modifying positons
            self.positions += self.velocities

            # inverting velocities depending on particle position
            self.velocities = np.where((self.positions < self.left_bound) | (self.best_positions > self.right_bound), -self.velocities, self.velocities)

            # Clipping positions
            self.positions = self.clip_position_higer(self.positions)
            self.positions = self.clip_positon_low(self.positions)

            # Updating personal record
            fittnes = np.apply_along_axis(objective_function, axis=1,  arr=self.positions)

            # Updating postions
            for j in range(self.num_pop):
                if fittnes[j] < best_fittnes[j]:
                    self.best_positions[j] = self.positions[j]

            best_fittnes = np.minimum(best_fittnes, fittnes)
            
            leader = np.argmin(best_fittnes)

            # Upadating result
            if global_minimum_value > objective_function(self.positions[leader]) and not np.isinf(best_fittnes[leader]):
                global_minimum_position = np.copy(self.positions[leader])
                global_minimum_value = objective_function(self.positions[leader])

            train_loss, test_loss = model.loss(global_minimum_position), model.test_loss(global_minimum_position)
            train_conv.append(train_loss)
            test_conv.append(test_loss)

            if i % 10 == 0:
                ax_convergence.set_title(f"Particle Swarm Optimization on iteration {i}")

                ax_convergence.plot(test_conv[-n:], label="Test")
                ax_convergence.plot(train_conv[-n:], label="Train")
                ax_convergence.legend()
                # Render the plot as an RGBA buffer
                canvas_convergence.draw()
                buf = canvas_convergence.buffer_rgba()

                # Create a PIL Image from the RGBA buffer
                image = Image.frombytes('RGBA', canvas_convergence.get_width_height(), buf, 'raw', 'RGBA')
                images_convergence.append(image)
                ax_convergence.cla()

                self.plot_regression(model, global_minimum_position, ax_regression)
                canvas_regression.draw()
                buf = canvas_regression.buffer_rgba()

                image = Image.frombytes('RGBA', canvas_regression.get_width_height(), buf, 'raw', 'RGBA')
                images_regression.append(image)
                ax_regression[0].cla()
                ax_regression[1].cla()

            if verbose:
                print("Ітерація %i\nЗначення %.3f\nТочка %s\n" % (i, global_minimum_value, global_minimum_position))

        # something like pause    
        images_convergence.extend([images_convergence[-1]] * max(100, num_iter // 25 + 1))

        images_convergence[0].save(f'{gif_name}_convergence.gif',
            save_all=True, append_images=images_convergence[1:], optimize=False, duration=10, loop=0)
        
        # something like pause    
        images_regression.extend([images_regression[-1]] * max(100, num_iter // 25 + 1))

        images_regression[0].save(f'{gif_name}_regression.gif',
            save_all=True, append_images=images_regression[1:], optimize=False, duration=10, loop=0)
        
        plt.close()
        plt.title(f"Convergence")

        plt.plot(range(n, num_iter), test_conv[n:], label="Test")
        plt.plot(range(n, num_iter), train_conv[n:], label="Train")
        plt.legend()
        plt.show()


        return np.squeeze(global_minimum_position)
    

class DE:
    def __init__(self, num_pop, num_dim, bounds):
        self.num_pop, self.num_dim = num_pop, num_dim
        self.left_bound, self.right_bound = bounds[:, 0], bounds[:, 1]
        self.population = np.random.rand(self.num_pop, self.num_dim) * (self.right_bound - self.left_bound) + self.left_bound

    def plot_regression(self, model, params, ax):
        left, right = min(model.X_train), max(model.X_train)
        r = np.linspace(left, right, 100)
        ax[0].scatter(model.X_train, model.y_train, label="Real data")
        ax[0].scatter(model.X_train, model.fit_function(model.X_train, params), label="Train")
        ax[0].plot(r, model.fit_function(r, params))
        ax[0].set_title("Training set")
        ax[0].legend()

        left, right = min(model.X_test), max(model.X_test)
        r = np.linspace(left, right, 100)
        ax[1].scatter(model.X_test, model.y_test, label="Real data")
        ax[1].scatter(model.X_test, model.fit_function(model.X_test, params), label="Test")
        ax[1].plot(r, model.fit_function(r, params))
        ax[1].set_title("Test set")
        ax[1].legend()

    def main_loop(self, num_iter, F, P, model, *, verbose=True, gif_name="DE"):
        best_res = np.inf
        best_pos = None

        train_conv, test_conv = [], []

        # plotting
        n = num_iter // 5
        fig_convergence, ax_convergence = plt.subplots()
        fig_regression, ax_regression = plt.subplots(1, 2, figsize=(10, 5))
        images_convergence = []
        images_regression = []
        canvas_convergence = FigureCanvasAgg(fig_convergence)
        canvas_regression = FigureCanvasAgg(fig_regression)


        for i in range(num_iter):
            for pos in range(self.num_pop):
                p1, p2, p3 = np.random.choice(self.num_pop, 3)

                while p1 == p2 or p2 == p3 or p1 == p3:
                    p1, p2, p3 = np.random.choice(self.num_pop, 3)

                x1, x2, x3 = self.population[[p1, p2, p3]]
                v = x1 + F * (x2 - x3)

                v = np.where(np.random.rand(self.num_dim, ) < P, self.population[pos], v)

                if model.loss(v) < model.loss(self.population[pos]):
                    self.population[pos] = v.copy()

            fitness = np.apply_along_axis(model.loss, 1, self.population)
            min_ind = fitness.argmin()

            if fitness[min_ind] < best_res:
                best_res = fitness[min_ind].copy()
                best_pos = self.population[min_ind].copy()

            train_loss, test_loss = model.loss(best_pos), model.test_loss(best_pos)
            train_conv.append(train_loss)
            test_conv.append(test_loss)

            if i % 10 == 0:
                ax_convergence.set_title(f"Differential Evolution Algorithm on iteration {i}")

                ax_convergence.plot(test_conv[-n:], label="Test")
                ax_convergence.plot(train_conv[-n:], label="Train")
                ax_convergence.legend()
                # Render the plot as an RGBA buffer
                canvas_convergence.draw()
                buf = canvas_convergence.buffer_rgba()

                # Create a PIL Image from the RGBA buffer
                image = Image.frombytes('RGBA', canvas_convergence.get_width_height(), buf, 'raw', 'RGBA')
                images_convergence.append(image)
                ax_convergence.cla()

                self.plot_regression(model, best_pos, ax_regression)
                canvas_regression.draw()
                buf = canvas_regression.buffer_rgba()

                image = Image.frombytes('RGBA', canvas_regression.get_width_height(), buf, 'raw', 'RGBA')
                images_regression.append(image)
                ax_regression[0].cla()
                ax_regression[1].cla()

            if verbose:
                print("Ітерація %i\nЗначення %.3f\nТочка %s\n" % (i + 1, best_res, best_pos))

        # something like pause    
        images_convergence.extend([images_convergence[-1]] * max(100, num_iter // 25 + 1))

        images_convergence[0].save(f'{gif_name}_convergence.gif',
            save_all=True, append_images=images_convergence[1:], optimize=False, duration=10, loop=0)
        
        # something like pause    
        images_regression.extend([images_regression[-1]] * max(100, num_iter // 25 + 1))

        images_regression[0].save(f'{gif_name}_regression.gif',
            save_all=True, append_images=images_regression[1:], optimize=False, duration=10, loop=0)
        
        plt.close()
        
        plt.title(f"Convergence")

        plt.plot(range(n, num_iter), test_conv[n:], label="Test")
        plt.plot(range(n, num_iter), train_conv[n:], label="Train")
        plt.legend()
        plt.show()

        return best_pos