""" Heuristic Optimization Algorithm developed in Python by Digvijay Gusain.
    This version of MVMO creates a surrogate model of the problem function to accelerate the optimization process.
    Usage: (Optimizing the Rosenbrock function constrained with a cubic and a line [Wikipedia])

        from MVMO import MVMO
        from MVMO import test_functions
        function = test_functions.rosen
        optimizer = MVMO(iterations=5000, num_mutation=3, population_size=10)

        def constraint(X):
            return True if X[0]**2 + X[1]**2 < 1 else False

        bds = [(-1.5,1.5), (-0.5,2.5)]
        constr = {'ineq':"(X[0] - 1)**3 - X[1] + 1",
                  'eq':"X[0]+X[1]-2",
                  'func':constraint}
        res, conv, sol, extras = optimizer.optimize(obj_fun=function, bounds=bds, constraints=constr)
        MVMO.plot(conv)

        Tip: num_mutation should be approximately 10% of Dimension and population_size ~ 10*num_mutation
    """
import numpy as np, pandas as pd, random
import time
from tqdm import tqdm
from pyDOE import lhs
import math
__version__ = "1.1.0"


class MVMO():

    def __init__(self, iterations=1000, num_mutation=1, scaling_factor=1, population_size=5,
                 logger=True, stop_iter_no_progresss=False, eps=1e-4,
                 speedup=False):
        # num_mutation can be variable.
        self.iterations = iterations
        self.num_mutation = num_mutation
        self.scaling_factor = scaling_factor
        self.population_size = population_size
        self.logger = logger
        self.stop_if_no_progress = stop_iter_no_progresss
        self.eps = eps
        self.speedup = speedup

    def mvmo(self, obj_fun, bounds, cons, x_initial, binary=[], integer=[]):

        # change - selection method
        current_mutation_position = 0

        # change - scaling_factor value consistent with the article
        scaling_factor = self.scaling_factor

        convergence = []
        min_b, max_b = np.asarray(bounds).T
        diff = max_b - min_b  # np.fabs(min_b - max_b)

        # problem dimension
        D = len(bounds)
        assert D >= self.num_mutation, "Number of mutations >= Problem dimension. Optimization cannot proceed."

        # create storage df
        solutions_d = []

        metrics_d = {}

        # change - variance initialization
        metrics_d['variance'] = np.asarray([1] * D)

        if type(x_initial) == np.ndarray:
            x0 = x_initial
            for item in x0:
                # denormalise solution
                x0_denorm = min_b + item * diff

                # evaluate initial fitness
                a = obj_fun(x0_denorm.tolist())
                # check if contraints are met
                sol_good = self.constraint_check(x0_denorm.tolist(), cons)

                if sol_good:
                    # change - redundant rounding
                    fitness = a
                    # fitness = round(a, 4)
                else:
                    fitness = 1e10  # penalty for constraint violation

                convergence.append(fitness)
                # fill the fitness dataframe with fitness value, solution vector, mean, shape, and d-factor

                solutions_d.append((fitness, tuple(item.tolist())))

        else:
            # initial population
            x0 = lhs(self.population_size, D).T
            for item in x0:
                if binary:
                    item[binary] = np.round(item[binary])
                if integer:
                    full_x = min_b + item * diff
                    full_x[integer] = np.round(full_x[integer])
                    item = (np.asarray(full_x) - min_b) / diff

                # denormalise solution
                x0_denorm = min_b + item * diff

                # evaluate initial fitness
                a = obj_fun(x0_denorm.tolist())
                # check if contraints are met
                sol_good = self.constraint_check(x0_denorm.tolist(), cons)

                if sol_good:
                    # change - redundant rounding
                    fitness = a
                    # fitness = round(a, 4)
                else:
                    fitness = 1e10  # penalty for constraint violation

                convergence.append(fitness)
                # fill the fitness dataframe with fitness value, solution vector, mean, shape, and d-factor

                solutions_d.append((fitness, tuple(item.tolist())))

        # change - proper setting best population
        solutions_d.sort()
        solutions_d = solutions_d[:self.population_size]
        worst_fitness = solutions_d[-1][0]

        # initial metric is set to 0.5 for mean
        scaling_factor_hist = []
        print_exit = False

        for i in tqdm(range(self.iterations), disable=not self.logger):
            # check for exit
            # change - commented stopping criterion - comparison just for specified iterations number
            # if self.stop_if_no_progress and i / self.iterations > 0.5 and np.var(convergence[-100:]) < self.eps:
            #     if not print_exit:
            #         print(f"Exiting at iteration {i} because optimizer couldn't improve solutions any longer.")
            #         print_exit = True
            #     continue

            # parent
            solutions_d.sort()

            x_parent = np.asarray(list(solutions_d[0][1]))
            # change - proper number of genes to mutation
            # num_mut = D if i < self.population_size else min(D, self.num_mutation)
            # if i > self.iterations / 3 and np.var(convergence[-500:]) < self.eps and self.speedup and bool(
            #         random.getrandbits(1)):
            #     num_mut = np.random.randint(1, num_mut)  # <--- new speedup

            # change - selection method consistent with article
            idxs = list(map(lambda x: x % D, [*range(current_mutation_position, current_mutation_position + self.num_mutation)]))
            current_mutation_position = current_mutation_position + self.num_mutation
            # idxs = np.random.choice(
            #     list(range(D)), num_mut, replace=False)

            # change - proper setting mean and variance
            # rand_mean = lhs(1, 1)[0][0]
            sol_d_tmp = pd.DataFrame.from_dict(dict(solutions_d), orient='columns')
            new_var = np.asarray([np.var(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))])
            new_var = np.round(new_var, 8)
            metrics_d['variance'] = np.where((np.isfinite(new_var)) & (new_var != 0), new_var, metrics_d['variance'])
            metrics_d['mean'] = [np.mean(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
            metrics_d['mean'] = np.round(metrics_d['mean'], 8)

            for idx in idxs:
                # mean
                # change - proper setting mean and variance
                # if len(solutions_d) > self.population_size + 1:
                #     x_bar = metrics_d['mean'][idx]
                #     var = metrics_d['variance'][idx]
                x_bar = metrics_d['mean'][idx]
                var = metrics_d['variance'][idx]
                # else:
                #     x_bar, var = rand_mean, 0.5

                # change - select a variable consistent with distribution mentioned in article
                xi_star = np.random.uniform(low=0, high=1, size=(1,))[0]
                # xi_star = np.clip(np.random.normal(0.5, 0.5, 1)[0], 0, 1)  # lhs(1,1)[0][0]

                # scaling factor can be variable. This affects converegnce so play with it.
                # maybe increase quadratically or someway with number of iterations.
                # when no improvement in solutions is observed, change it back to one for more explorstion\
                #                scaling_factor = 1 + (i/self.iterations)*(19)

                # change - constant scaling_factor with no speed up - consistent with the article
                # scaling_factor = 1 + (i ** 2.4)  # <--- new scaling factor
                # if i > 500 and np.var(convergence[-50:]) < self.eps and bool(random.getrandbits(1)):
                #     scaling_factor = 1.  # <--- new scaling. previously was 2.

                s_old = -np.log(var) * scaling_factor

                # change - mapping function fixed, previous version was not consistent with the article
                # this 0.5 can also be adaptive
                hx = x_bar * (1 - math.exp(-1 * xi_star * s_old)) + (1 - x_bar) * math.exp((xi_star - 1) * s_old)
                h0 = x_bar * (1 - math.exp(-1 * 0 * s_old)) + (1 - x_bar) * math.exp((0 - 1) * s_old)
                h1 = x_bar * (1 - math.exp(-1 * 1 * s_old)) + (1 - x_bar) * math.exp((1 - 1) * s_old)
                xi_new = hx + (1 - h1 + h0) * xi_star - h0
                # if xi_star < 0.5:
                #     s_new = s_old / (1 - x_bar)
                #     hm = x_bar - x_bar / (0.5 * s_new + 1)
                #     hf = x_bar * (1 - np.exp(-xi_star * s_new))
                #     hc = (x_bar - hm) * 5 * xi_star
                #     xi_new = hf + hc
                # else:
                #     s_new = s_old / x_bar
                #     hm = (1 - x_bar) / (0.5 * s_new + 1)
                #     hb = (1 - x_bar) / ((1 - xi_star) * s_new + 1) + x_bar
                #     hc = hm * 5 * (1 - xi_star)
                #     xi_new = hb - hc
                x_parent[idx] = xi_new
            scaling_factor_hist.append(scaling_factor)

            if binary:
                x_parent[binary] = np.round(x_parent[binary])

            if integer:
                full_x = min_b + x_parent * diff
                full_x[integer] = np.round(full_x[integer])
                x_parent = (np.asarray(full_x) - min_b) / diff

            x_denorm = min_b + x_parent * diff

            tmp = x_denorm
            # tmp = x_denorm.tolist()

            a = obj_fun(tmp)
            sol_good = self.constraint_check(x_denorm, cons)

            if sol_good:
                # change - redundant rounding
                fitness = a
                # fitness = round(a, 4)
            else:
                fitness = 1e10  # penalty for constraint violation

            # change - fixed condition for adding new individual
            if len(solutions_d) < self.population_size:
            # if len(solutions_d) < self.population_size + 1:
                solutions_d.append((fitness, tuple(x_parent.tolist())))
                solutions_d.sort()
                convergence.append(solutions_d[0][0])

                # change - proper mean and variance setting
                # sol_d_tmp = pd.DataFrame.from_dict(dict(solutions_d), orient='columns')
                # metrics_d['variance'] = [
                #     np.var(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                # metrics_d['mean'] = [
                #     np.mean(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]

            else:
                if fitness < worst_fitness:
                    solutions_d.append((fitness, tuple(x_parent.tolist())))
                    solutions_d.sort()
                    solutions_d.pop(-1)
                    convergence.append(solutions_d[0][0])

                    # change - proper mean and variance setting
                    # sol_d_tmp = pd.DataFrame.from_dict(dict(solutions_d), orient='columns')
                    #
                    # metrics_d['variance'] = [
                    #     np.var(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                    # metrics_d['mean'] = [
                    #     np.mean(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]

                    worst_fitness = solutions_d[-1][0]
                else:
                    convergence.append(convergence[-1])

        solutions_d.sort()
        res = min_b + \
              np.asarray(list(solutions_d[0][1])) * diff
        # change - redundant rounding
        # res = [round(x, 7) for x in res]
        final_of = obj_fun(res)

        res_dict_final = {
            'objective': final_of,
            'x': res,
            'convergence': convergence,
            'register': pd.DataFrame.from_dict(dict(solutions_d), orient='index'),
            'metrics': metrics_d,
            'scaling_factors': scaling_factor_hist
        }
        return res_dict_final

    def constraint_check(self, solution, constraints):
        if len(constraints) == 0:
            return True
        else:
            X = solution
            for key, value in constraints.items():
                if key != 'func':
                    v = eval(value)
                    if key == 'ineq' and v >= 0:
                        return False
                    elif key == 'eq' and v != 0:
                        return False
                    else:
                        return True
                else:
                    return value(X)

    def optimize(self, obj_fun, bounds, constraints={}, x0=False, binary=[], integer=[]):
        t1 = time.time()
        self.res = self.mvmo(obj_fun=obj_fun, bounds=bounds, cons=constraints,
                             x_initial=x0, binary=binary, integer=integer)
        t2 = time.time()
        if self.logger:
            sep = '*' * len(list(f"Optimal Solution found in {round(t2 - t1, 2)}s"))
            print("\n")
            print(sep)
            print(f"Optimal Solution found in {round(t2 - t1, 2)}s")
            print(sep)
            print(f"\nFinal Objective Function Value: {self.res['objective']}.")

        return self.res

    def plot(conv):
        import matplotlib.pyplot as plt
        plt.plot(conv, "C2", linewidth=1, label='OF value')
        plt.ylabel("Objective Function Fitness")
        plt.xlabel("Iterations")
        plt.title("Convergence Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()


class test_functions():
    def rosen(X):
        x = X[0]
        y = X[1]
        a = 1. - x
        b = y - x * x
        return a * a + b * b * 100.

    def obf(x):
        return sum(np.asarray(x) ** 2) + 2

    def booth(x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    def matyas(x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

    def himm(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def easom(x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-1 * ((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
