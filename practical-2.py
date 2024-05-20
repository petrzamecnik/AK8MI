import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import random
import time
import sys
import math
import pprint
import matplotlib.pyplot as plt
import os
import pandas as pd


def generate_mckp_instance(num_classes, num_items_per_class=3):
    items = []
    for class_id in range(1, num_classes + 1):
        for item_id in range(1, num_items_per_class + 1):
            weight = random.randint(1, 50)
            price = random.randint(1, 50)
            item = {'class_id': class_id, 'item_id': item_id, 'weight': weight, 'price': price}
            items.append(item)
    return items


def get_capacity_mckp(num_classes):
    if 3 <= num_classes <= 5:
        return 100
    elif 6 <= num_classes <= 10:
        return 200
    else:
        return 300


# Brute Force
def evaluate_solution(solution, weights, prices, capacity):
    total_weight = 0
    total_price = 0
    for class_idx, item_idx in enumerate(solution):
        total_weight += weights[class_idx][item_idx]
        total_price += prices[class_idx][item_idx]
    if total_weight > capacity:
        return 0
    return total_price


def brute_force_mckp(items, num_classes, capacity):
    best_solution = None
    best_price = 0

    # Generate all possible combinations of items from different classes
    class_items = []
    for i in range(1, num_classes + 1):
        class_items.append([item for item in items if item['class_id'] == i])

    for combination in product(*class_items):
        total_weight = sum(item['weight'] for item in combination)
        total_price = sum(item['price'] for item in combination)

        if total_weight <= capacity and total_price > best_price:
            best_solution = combination
            best_price = total_price

    return best_solution, best_price


def test_brute_force_mckp(items, capacity, num_classes):
    start_time = time.time()
    best_solution, best_price = brute_force_mckp(items, num_classes, capacity)
    end_time = time.time()

    return best_solution, best_price, end_time - start_time


# Simulated annealing
def generate_initial_solution(num_classes, num_items_per_class):
    return [random.randint(0, num_items_per_class - 1) for _ in range(num_classes)]


def generate_neighbor_solution(solution, num_classes, num_items_per_class):
    neighbor = solution.copy()
    class_to_change = random.randint(0, num_classes - 1)
    new_item = random.randint(0, num_items_per_class - 1)
    while new_item == neighbor[class_to_change]:
        new_item = random.randint(0, num_items_per_class - 1)
    neighbor[class_to_change] = new_item
    return neighbor


def simulated_annealing_mckp(items, num_classes, num_items_per_class, capacity, initial_temp, final_temp, cooling_rate,
                             max_iterations, output_dir):
    max_fes_brute_force = 3 ** num_classes  # Maximum number of combinations in brute force

    current_solution = generate_initial_solution(num_classes, num_items_per_class)
    current_price = evaluate_solution_mckp(current_solution, items, num_classes, num_items_per_class, capacity)
    best_solution = current_solution
    best_price = current_price

    temp = initial_temp
    iteration = 0
    best_prices = []

    while temp > final_temp and iteration < max_iterations and iteration < max_fes_brute_force:
        neighbor_solution = generate_neighbor_solution(current_solution, num_classes, num_items_per_class)
        neighbor_price = evaluate_solution_mckp(neighbor_solution, items, num_classes, num_items_per_class, capacity)

        if neighbor_price > current_price:
            current_solution = neighbor_solution
            current_price = neighbor_price
            if current_price > best_price:
                best_solution = current_solution
                best_price = current_price
        else:
            acceptance_probability = math.exp((neighbor_price - current_price) / temp)
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_price = neighbor_price

        best_prices.append(best_price)
        temp *= cooling_rate
        iteration += 1

    plt.plot(best_prices)
    plt.title("Convergence Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Best Price")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "convergence_graph.png")
    plt.savefig(output_path)
    plt.close()

    return best_solution, best_price


def evaluate_solution_mckp(solution, items, num_classes, num_items_per_class, capacity):
    total_weight = 0
    total_price = 0
    for class_idx, item_idx in enumerate(solution):
        item = items[class_idx * num_items_per_class + item_idx]
        total_weight += item['weight']
        total_price += item['price']
    if total_weight > capacity:
        return 0
    return total_price


def test_simulated_annealing_mckp(items, capacity, num_classes, num_items_per_class=3, output_dir="practical-output"):
    initial_temp = 10000
    final_temp = 0.001
    cooling_rate = 0.92
    max_iterations = 1000

    start_time = time.time()
    best_solution_indices, best_price = simulated_annealing_mckp(items, num_classes, num_items_per_class, capacity,
                                                                 initial_temp, final_temp, cooling_rate, max_iterations,
                                                                 output_dir)
    end_time = time.time()
    best_solution_items = [items[class_idx * num_items_per_class + item_idx] for class_idx, item_idx in
                           enumerate(best_solution_indices)]
    total_weight = sum(item['weight'] for item in best_solution_items)

    print("Best solution:")
    pprint.pprint(best_solution_items)
    print("Best price:", best_price)
    print("Time taken: {:.3f} seconds".format(end_time - start_time))

    return best_solution_indices, best_price, end_time - start_time, total_weight


def run_multiple_iterations(algorithm, num_runs, *args):
    results = []
    for _ in range(num_runs):
        best_solution, best_price, time_taken, total_weight = algorithm(*args)
        results.append({
            'best_price': best_price,
            'time_taken': time_taken,
            'total_weight': total_weight
        })
    return results


def create_csv_from_results(results, best_price_brute_force, max_weight_possible, output_csv):
    for result in results:
        result['time_taken'] = f"{result['time_taken']:.3f}"

    df = pd.DataFrame(results)
    df['best_price_possible'] = best_price_brute_force
    df['max_weight_possible'] = max_weight_possible
    df['number_of_run'] = df.index + 1
    df = df[['number_of_run', 'time_taken', 'best_price', 'best_price_possible', 'total_weight', 'max_weight_possible']]
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def run_brute_force_within_one_hour():
    num_classes = 1
    num_items_per_class = 3
    total_time = 0
    max_time = 3600  # 1 hour in seconds

    while total_time < max_time:
        items = generate_mckp_instance(num_classes, num_items_per_class)
        capacity = get_capacity_mckp(num_classes)

        start_time = time.time()
        best_solution, best_price = brute_force_mckp(items, num_classes, capacity)
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_time += elapsed_time

        print(f"Number of classes: {num_classes}")
        print(f"Time taken: {elapsed_time:.3f} seconds")
        print(f"Total accumulated time: {total_time:.3f} seconds")
        print("-" * 50)

        if total_time < max_time:
            num_classes += 1
        else:
            break

    print("Finished. Maximum time reached or exceeded.")


if __name__ == '__main__':
    number_of_classes = 10
    num_items_per_class = 3

    items = generate_mckp_instance(number_of_classes)
    capacity = get_capacity_mckp(number_of_classes)
    output_dir = "practical-output"
    output_csv = os.path.join(output_dir, "sa_results.csv")

    best_solution_brute_force, best_price_brute_force, _ = test_brute_force_mckp(items, capacity, number_of_classes)

    print("-" * 50)
    print("Simulated Annealing Multiple Runs:")
    results = run_multiple_iterations(test_simulated_annealing_mckp, 30, items, capacity, number_of_classes,
                                      num_items_per_class, output_dir)

    create_csv_from_results(results, best_price_brute_force, capacity, output_csv)

    run_brute_force_within_one_hour()