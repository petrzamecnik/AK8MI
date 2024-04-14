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
def random_search(test_function, iterations, bounds, dimension):
    def generate_random_solution(_bounds, _dimension):
        while True:
            solution = np.random.uniform(*_bounds, _dimension)
            if np.all(_bounds[0] <= solution) and np.all(solution <= _bounds[1]):
                return solution

    random_solution = generate_random_solution(bounds, dimension)
    best_solution = random_solution
    best_value = test_function(best_solution)
    best_values = [best_value]

    for i in range(iterations - 1):
        random_solution = generate_random_solution(bounds, dimension)
        value = test_function(random_solution)
        best_values.append(value)

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


# Configuration
bounds_dejong = (-5, 5)
bounds_schweffel = (-500, 500)

# Implementations
best_solution_dejong1, best_value_dejong1, best_values_dejong1 = random_search(dejong1, 5, bounds_dejong, 5)
statistics_dejong1 = calculate_statistics(best_values_dejong1)

pprint.pprint(statistics_dejong1)
