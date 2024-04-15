import numpy as np
import pprint
import matplotlib.pyplot as plt


# Test Functions
def dejong1(x):
    x = np.array(x)
    return np.sum(x ** 2)


def dejong2(x):
    # f(x⃗) = Σ (x_i^2 - 10 * cos(2π * x_i) + 10) pro i = 1, 2, ..., n

    x = np.array(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def Schweffel(x):
    # (x⃗) = -Σ (x_i * sin(sqrt(|x_i|))) pro i = 1, 2, ..., n
    x = np.array(x)
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Main Functions
def random_search(test_function, max_fes, bounds, dimension):
    def generate_random_solution(_bounds, _dimension):
        while True:
            solution = np.random.uniform(*_bounds, _dimension)
            if np.all(_bounds[0] <= solution) and np.all(solution <= _bounds[1]):
                return solution

    random_solution = generate_random_solution(bounds, dimension)
    best_solution = random_solution
    best_value = test_function(best_solution)
    best_values = [best_value]
    num_fes = 1

    while num_fes < max_fes:
        random_solution = generate_random_solution(bounds, dimension)
        value = test_function(random_solution)
        best_values.append(value)
        num_fes += 1

        if value < best_value:
            best_value = value
            best_solution = random_solution

    return best_solution, best_value, best_values

def simulated_annealing(_test_function, _iterations, _bounds, _dimension, _temp_start, _temp_end):
    return NotImplemented


# Statistics
def calculate_statistics(best_values):
    best_values = np.array(best_values)

    return (
        np.min(best_values),
        np.max(best_values),
        np.mean(best_values),
        np.median(best_values),
        np.std(best_values)
    )


def plot_all_runs(title, best_values):
    plt.figure()
    # best_values_sorted = np.sort(best_values)

    for best_value in best_values:
        plt.plot(best_value)

    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Best Value')
    plt.show()


def plot_average_convergence(title, best_values):
    average_best_values = np.mean(best_values, axis=0)
    average_best_values_sorted = np.sort(average_best_values)[::-1]

    plt.figure()
    plt.plot(average_best_values_sorted)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Average Best Value')
    plt.show()


# Configuration
bounds_dejong = (-5, 5)
bounds_schweffel = (-500, 500)
iterations = 100


# Implementations
def experiment_dejong1():
    num_runs = 30
    all_runs = []

    for i in range(num_runs):
        _, _, best_values = random_search(dejong1, iterations, bounds_dejong, 5)
        all_runs.append(best_values)

    min_val, max_val, mean_val, median_val, std_val = calculate_statistics(best_values)
    print(
        f"Minimum: {min_val}, Maximum: {max_val}, Mean: {mean_val}, Median: {median_val}, Standard Deviation: {std_val}")

    plot_all_runs("All Runs Dejong1", all_runs)
    plot_average_convergence("Average Convergence", all_runs)


experiment_dejong1()
