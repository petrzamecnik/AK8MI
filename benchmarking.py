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
def random_search(_test_function, _iterations, _bounds, _dimension):
    def generate_random_solution(_bounds, _dimension):
        while True:
            solution = np.random.uniform(*_bounds, _dimension)
            if np.all(_bounds[0] <= solution) and np.all(solution <= _bounds[1]):
                return solution

    random_solution = generate_random_solution(_bounds, _dimension)
    _best_solution = random_solution
    _best_value = _test_function(_best_solution)

    for _ in range(_iterations - 1):
        random_solution = generate_random_solution(_bounds, _dimension)
        value = _test_function(random_solution)

        if value < _best_value:
            _best_value = value
            _best_solution = random_solution

    return _best_solution, _best_value


def simulated_annealing(_test_function, _iterations, _bounds, _dimension, _temp_start, _temp_end):
    return NotImplemented


# Configuration
bounds_dejong = (-5, 5)
bounds_schweffel = (-500, 500)

# Implementations
best_solution, best_value = random_search(dejong1, 100, bounds_dejong, 5)
print("best_solution:", best_solution)
print("best_value:", best_value)
