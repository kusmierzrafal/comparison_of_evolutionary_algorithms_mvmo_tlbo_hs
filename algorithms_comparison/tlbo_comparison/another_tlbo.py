# python implementation of Teaching learning based optimization (TLBO)
# minimizing rastrigin and sphere function

import random
import math  # cos() for Rastrigin
import copy  # array-copying convenience
import sys  # max float
import numpy as np  # just for levy testing

# -------fitness functions---------

# rastrigin function
def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


# sphere function
def fitness_sphere(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi);
    return fitness_value;


# rosenbrock function
def fitness_rosenbrock(position):
    fitness_value = 0.0
    for i in range(len(position)-1):
        x1 = position[i]
        x2 = position[i+1]
        fitness_value += 100 * (x1 ** 2 - x2) ** 2 + (x2 - 1) ** 2
    return fitness_value


# zakharov function
def fitness_zakharov(position):
    part1 = 0.0
    part2 = 0.0
    for i in range(len(position)):
        x1 = position[i]
        part1 += x1**2
        part2 += 0.5 * x1
    fitness_value = part1 + part2 ** 2 + part2 ** 4
    return fitness_value


def fitness_levy(position):
    position = np.asarray(position)
    position_w = 1 + (position - 1) / 4
    return (
        np.sin(math.pi * position_w[0]) ** 2
        + (position_w[-1] - 1) ** 2 * (1 + np.sin(2 * math.pi * position_w[-1]) ** 2)
        + sum(
            [
                (wi - 1) ** 2 * (1 + 10 * np.sin(math.pi * wi + 1) ** 2)
                for wi in position_w[:-1]
            ]
        )
    )

# -------------------------


# Student class
class Student:
    def __init__(self, fitness, position):

        self.position = position

        # compute the fitness of student
        self.fitness = fitness(self.position)


# Teaching learning based optimization
def tlbo(fitness, max_iter, population, boundaries):
    minx, maxx = boundaries
    n = len(population)
    dim = len(population[0])

    # create n random students
    classroom = [Student(fitness, ind.tolist()) for ind in population]

    # compute the value of best_position and best_fitness in the classroom
    Xbest = [0.0 for i in range(dim)]
    Fbest = sys.float_info.max

    for i in range(n):  # check each Student
        if classroom[i].fitness < Fbest:
            Fbest = classroom[i].fitness
            Xbest = copy.copy(classroom[i].position)

    # main loop of tlbo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
            # print("Iter = " + str(Iter) + " best fitness = %.3f" % Fbest)

        # for each student of classroom
        for i in range(n):

            ### Teaching phase of ith student

            # compute the mean of all the students in the class
            Xmean = [0.0 for i in range(dim)]
            for k in range(n):
                for j in range(dim):
                    Xmean[j] += classroom[k].position[j]

            for j in range(dim):
                Xmean[j] /= n;

            # initialize new solution
            Xnew = [0.0 for i in range(dim)]

            # teaching factor (TF)
            # either 1 or 2 ( randomly chosen)
            TF = random.randint(1, 2)

            # best student of the class is teacher
            Xteacher = Xbest

            # compute new solution
            for j in range(dim):
                Xnew[j] = classroom[i].position[j] + random.random() * (Xteacher[j] - TF * Xmean[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

            ### learning phase of ith student

            # randomly choose a solution from classroom
            # chosen solution should not be ith student
            p = random.randint(0, n - 1)
            while (p == i):
                p = random.randint(0, n - 1)

            # partner solution
            Xpartner = classroom[p]

            Xnew = [0.0 for i in range(dim)]
            if (classroom[i].fitness < Xpartner.fitness):
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] + random.random() * (
                                classroom[i].position[j] - Xpartner.position[j])
            else:
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] - random.random() * (
                                classroom[i].position[j] - Xpartner.position[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

        Iter += 1
    # end-while

    # return best student from classroom
    return Xbest


# end pso


# ----------------------------
# Driver code for rastrigin function
if __name__ == '__main__':
    print("\nBegin  teaching learning based optimization on rastrigin function\n")
    dim = 3
    fitness = fitness_rastrigin

    print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
    print("Function has known min = 0.0 at (", end="")
    for i in range(dim - 1):
        print("0, ", end="")
    print("0)")

    num_particles = 50
    max_iter = 100

    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter    = " + str(max_iter))
    print("\nStarting TLBO algorithm\n")

    best_position = tlbo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

    print("\nTLBO completed\n")
    print("\nBest Student found:")
    print(["%.6f" % best_position[k] for k in range(dim)])
    fitness_value = fitness(best_position)
    print("fitness of best Student = %.6f" % fitness_value)

    print("\nEnd TLBO for rastrigin function\n")

    print()
    print()

    # Driver code for Sphere function
    print("\nBegin teaching learning based optimization on sphere function\n")
    dim = 3
    fitness = fitness_sphere

    print("Goal is to minimize sphere function in " + str(dim) + " variables")
    print("Function has known min = 0.0 at (", end="")
    for i in range(dim - 1):
        print("0, ", end="")
    print("0)")

    num_particles = 50
    max_iter = 100

    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter    = " + str(max_iter))
    print("\nStarting TLBO algorithm\n")

    best_position = tlbo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

    print("\nTLBO completed\n")
    print("\nBest Student found:")
    print(["%.6f" % best_position[k] for k in range(dim)])
    fitness_value = fitness(best_position)
    print("fitness of best Student = %.6f" % fitness_value)

    print("\nEnd TLBO for sphere function\n")