import numpy as np
import matplotlib.pyplot as plt


# Test Functions
def dejong1(x):
    x = np.array(x)
    return np.sum(x ** 2)


def dejong2(x):
    # f(x⃗) = Σ (x_i^2 - 10 * cos(2π * x_i) + 10) pro i = 1, 2, ..., n
    x = np.array(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def schweffel(x):
    # (x⃗) = -Σ (x_i * sin(sqrt(|x_i|))) pro i = 1, 2, ..., n
    x = np.array(x)
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Main Functions
def compute_stats(best_values):
    return {
        'min': np.min(best_values),
        'max': np.max(best_values),
        'mean': np.mean(best_values),
        'median': np.median(best_values),
        'std': np.std(best_values)
    }


def random_search(function, bounds, max_iter):
    best_solution = None
    best_value = np.inf
    best_values = []

    for i in range(max_iter):
        solution = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
        value = function(solution)

        if value < best_value:
            best_solution = solution
            best_value = value

        best_values.append(best_value)

    return best_solution, best_value, best_values


def simulated_annealing(function, bounds, max_iter, initial_temp, cooling_rate):
    current_solution = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
    current_value = function(current_solution)
    best_solution = None
    best_value = current_value
    temperature = initial_temp



# Plots
def plot_average_convergence(func_name, dimensions, algo_name, all_run_results):
    max_iter = len(all_run_results[0])
    average_best_values = [np.mean([run_results[i] for run_results in all_run_results]) for i in range(max_iter)]

    plt.figure(figsize=(12, 4))
    plt.plot(range(max_iter), average_best_values)
    plt.title(f'Average Best Solution - {algo_name} - {func_name} (Dimensions = {dimensions})')
    plt.xlabel('Iterations')
    plt.ylabel('Average Best Function Value')
    plt.tight_layout()

    plt.show()
    plt.close()


def plot_all_convergences(func_name, dimensions, algo_name, all_run_results):
    max_iter = len(all_run_results[0])
    num_runs = len(all_run_results)

    plt.figure(figsize=(12, 6))
    for i in range(num_runs):
        plt.plot(range(max_iter), all_run_results[i], label=f'Run {i + 1}')

    plt.title(f'All Convergences - {algo_name} - {func_name} (Dimensions = {dimensions})')
    plt.xlabel('Iterations')
    plt.ylabel('Best Function Value')
    plt.tight_layout()

    plt.show()
    plt.close()


# Config
dimensions = (5, 10)
max_fes = 10000
num_runs = 30

bounds_dejong = np.array([-5, 5])
bounds_schweffel = np.array([-500, 500])


# Experiments
def experiment_dejong1():
    results = []
    for dimension in dimensions:
        bounds = np.tile(bounds_dejong, dimension)
        run_results = []
        for i in range(num_runs):
            _, _, best_values = random_search(dejong1, bounds, max_fes)
            run_results.append(best_values)
        results.append(run_results)
        print(f"Dejong 1 (Dimensions = {dimension})")
        print(compute_stats([run[-1] for run in run_results]))

        plot_average_convergence('dejong1', dimension, 'Random Search', run_results)
        plot_all_convergences('dejong1', dimension, 'Random Search', run_results)

    return results


def experiment_dejong2():
    results = []
    for dimension in dimensions:
        bounds = np.tile(bounds_dejong, dimension)
        run_results = []
        for i in range(num_runs):
            _, _, best_values = random_search(dejong2, bounds, max_fes)
            run_results.append(best_values)
        results.append(run_results)
        print(f"Dejong 2 (Dimensions = {dimension})")
        print(compute_stats([run[-1] for run in run_results]))

        plot_average_convergence('dejong2', dimension, 'Random Search', run_results)
        plot_all_convergences('dejong2', dimension, 'Random Search', run_results)

    return results


def experiment_schweffel():
    results = []
    for dimension in dimensions:
        bounds = np.tile(bounds_dejong, dimension)
        run_results = []
        for i in range(num_runs):
            _, _, best_values = random_search(schweffel, bounds, max_fes)
            run_results.append(best_values)
        results.append(run_results)
        print(f"Schweffel (Dimensions = {dimension})")
        print(compute_stats([run[-1] for run in run_results]))

        plot_average_convergence('schweffel', dimension, 'Random Search', run_results)
        plot_all_convergences('schweffel', dimension, 'Random Search', run_results)

    return results


# Runs
# experiment_dejong1()
# experiment_dejong2()
# experiment_schweffel()
