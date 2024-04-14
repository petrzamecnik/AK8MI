import numpy as np


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
    random_solution = np.random.uniform(*bounds, dimension)
    _best_solution = random_solution
    _best_value = np.inf

    for i in range(iterations):
        random_solution = np.random.uniform(*bounds, dimension)
        value = test_function(random_solution)

        if value < _best_value:
            _best_value = value
            _best_solution = random_solution

    return _best_solution, _best_value


# Configuration
bounds_dejong = (-5, 5)
bounds_schweffel = (-500, 500)

# Implementations
best_solution, best_value = random_search(dejong1, 100, bounds_dejong, 5)
print("best_solution:", best_solution)
print("best_value:", best_value)
