import numpy as np
from itertools import product
import random
import time
import sys
import math

# Config
num_classes = 5
num_items_per_class = 5


# Main Functions
def generate_mckp_instance(num_classes, num_items_per_class):
    weights = np.random.randint(1, 51, size=(num_classes, num_items_per_class))
    prices = np.random.randint(1, 51, size=(num_classes, num_items_per_class))
    return weights, prices


def get_knapsack_capacity(num_classes):
    if 3 <= num_classes <= 5:
        return 100
    elif 6 <= num_classes <= 10:
        return 200
    else:
        return 300


def evaluate_solution(solution, weights, prices, capacity):
    total_weight = 0
    total_price = 0
    for class_idx, item_idx in enumerate(solution):
        total_weight += weights[class_idx][item_idx]
        total_price += prices[class_idx][item_idx]
    if total_weight > capacity:
        return 0  # Nepřijatelné řešení, překročena kapacita
    return total_price


def brute_force(weights, prices, capacity):
    start_time = time.time()
    best_solution = None
    best_price = 0
    for solution in product(*[range(weights.shape[1]) for _ in range(weights.shape[0])]):
        price = evaluate_solution(solution, weights, prices, capacity)
        if price > best_price:
            best_price = price
            best_solution = solution
    end_time = time.time()
    elapsed_time = end_time - start_time
    return best_solution, best_price, elapsed_time


def generate_initial_solution(num_classes):
    return [random.randint(0, num_items_per_class - 1) for _ in range(num_classes)]


def generate_neighbor(solution):
    neighbor = solution.copy()
    class_idx = random.randint(0, num_classes - 1)
    neighbor[class_idx] = random.randint(0, num_items_per_class - 1)
    return neighbor


def simulated_annealing(weights, prices, capacity, max_iterations, initial_temp, cooling_rate):
    start_time = time.time()
    current_solution = generate_initial_solution(num_classes)
    current_price = evaluate_solution(current_solution, weights, prices, capacity)
    best_solution = current_solution
    best_price = current_price
    temperature = initial_temp

    for iteration in range(max_iterations):
        neighbor = generate_neighbor(current_solution)
        neighbor_price = evaluate_solution(neighbor, weights, prices, capacity)

        if neighbor_price > current_price:
            current_solution = neighbor
            current_price = neighbor_price
        else:
            delta = neighbor_price - current_price
            if random.uniform(0, 1) < math.exp(delta / temperature):
                current_solution = neighbor
                current_price = neighbor_price

        if current_price > best_price:
            best_solution = current_solution
            best_price = current_price

        temperature *= cooling_rate

    end_time = time.time()
    elapsed_time = end_time - start_time
    return best_solution, best_price, elapsed_time


# Run
weights, prices = generate_mckp_instance(num_classes, num_items_per_class)
capacity = get_knapsack_capacity(num_classes)

# Hrubá síla
brute_force_solution, brute_force_price, brute_force_time = brute_force(weights, prices, capacity)
print("Brute Force Solution:", brute_force_solution)
print("Brute Force Price:", brute_force_price)
print(f"Brute Force Time: {brute_force_time:.2f} s")

# Simulované žíhání
sa_solution, sa_price, sa_time = simulated_annealing(weights, prices, capacity, max_iterations=10000, initial_temp=1000,
                                                     cooling_rate=0.99)
print("Simulated Annealing Solution:", sa_solution)
print("Simulated Annealing Price:", sa_price)
print(f"Simulated Annealing Time: {sa_time:.2f} s")

# Hledání maximálního počtu tříd řešitelného do 1 hodiny
MAX_TIME = 3600  # 1 hodina v sekundách

print("\nHledání maximálního počtu tříd řešitelného do 1 hodiny...")

num_classes = 3
while True:
    weights, prices = generate_mckp_instance(num_classes, num_items_per_class)
    capacity = get_knapsack_capacity(num_classes)

    print(f"\nPočet tříd: {num_classes}")

    brute_force_solution, brute_force_price, brute_force_time = brute_force(weights, prices, capacity)
    print(f"Brute Force: Cena = {brute_force_price}, Čas = {brute_force_time:.2f} s")

    if brute_force_time > MAX_TIME:
        print(f"Maximální počet tříd řešitelných do 1 hodiny je {num_classes - 1}")
        sys.exit(0)

    sa_solution, sa_price, sa_time = simulated_annealing(weights, prices, capacity, max_iterations=10000,
                                                         initial_temp=1000, cooling_rate=0.99)
    print(f"Simulated Annealing: Cena = {sa_price}, Čas = {sa_time:.2f} s")

    num_classes += 1
