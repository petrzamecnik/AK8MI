import math
import random
import time
import matplotlib.pyplot as plt


def generate_mckp_instance(num_classes, items_per_class=3):
    """Generuje náhodnou instanci problému batohu s vícenásobnou volbou."""
    prices = []
    volumes = []
    for i in range(num_classes):
        class_prices = [random.randint(1, 50) for _ in range(items_per_class)]
        class_volumes = [random.randint(1, 50) for _ in range(items_per_class)]
        prices.append(class_prices)
        volumes.append(class_volumes)

    capacity = 100 + 100 * ((num_classes - 3) // 4)  # Výpočet kapacity
    return prices, volumes, capacity


def objective_function(solution, prices, volumes, capacity):
    """Vypočítá hodnotu řešení a kontroluje, zda je přípustné."""
    total_price = 0
    total_volume = 0
    for i, item_idx in enumerate(solution):
        total_price += prices[i][item_idx]
        total_volume += volumes[i][item_idx]

    if total_volume > capacity:
        return 0  # Nepřípustné řešení
    return total_price


def brute_force_mckp(prices, volumes, capacity):
    """Řeší problém batohu s vícenásobnou volbou hrubou silou."""
    best_price = 0
    best_solution = None
    num_combinations = 1
    for c in prices:
        num_combinations *= len(c)

    for i in range(num_combinations):
        solution = []
        temp_i = i
        for j in range(len(prices)):
            solution.append(temp_i % len(prices[j]))
            temp_i //= len(prices[j])
        price = objective_function(solution, prices, volumes, capacity)
        if price > best_price:
            best_price = price
            best_solution = solution
    return best_price, best_solution


def simulated_annealing_mckp(prices, volumes, capacity, initial_temp=1000, cooling_rate=0.95, iterations=1000):
    """Řeší problém batohu s vícenásobnou volbou pomocí simulovaného žíhání."""
    current_solution = [random.randrange(len(c)) for c in prices]
    best_solution = current_solution.copy()
    best_price = objective_function(best_solution, prices, volumes, capacity)
    current_temp = initial_temp
    best_prices = [best_price]  # Pro ukládání průběhu nejlepší ceny

    for _ in range(iterations):
        new_solution = current_solution.copy()
        idx_to_change = random.randrange(len(new_solution))
        new_solution[idx_to_change] = random.randrange(len(prices[idx_to_change]))
        new_price = objective_function(new_solution, prices, volumes, capacity)

        if new_price > best_price:
            best_price = new_price
            best_solution = new_solution

        delta = new_price - objective_function(current_solution, prices, volumes, capacity)
        if delta > 0 or random.random() < math.exp(delta / current_temp):
            current_solution = new_solution

        current_temp *= cooling_rate
        best_prices.append(best_price)  # Uložení nejlepší ceny

    return best_price, best_solution, best_prices


# Generování instance problému
num_classes = 15  # Počet tříd předmětů
prices, volumes, capacity = generate_mckp_instance(num_classes)

# Řešení hrubou silou
start_time = time.time()
brute_force_price, brute_force_solution = brute_force_mckp(prices, volumes, capacity)
brute_force_time = time.time() - start_time

# Řešení simulovaným žíháním
start_time = time.time()
sa_price, sa_solution, best_prices = simulated_annealing_mckp(prices, volumes, capacity)
sa_time = time.time() - start_time

print("\nProblém batohu s vícenásobnou volbou:")
print("Počet tříd:", num_classes)
print("Kapacita batohu:", capacity)

# Ukázka menší instance problému
print("\nUkázka menší instance (5 tříd):")
for i in range(5):
    print(f"Třída {i + 1}:")
    for j in range(3):
        print(f"- ID {j + 1}, Objem {volumes[i][j]}, Cena {prices[i][j]}")

print("\nŘešení hrubou silou:")
print("- Nejlepší cena:", brute_force_price)
print("- Nejlepší řešení:", brute_force_solution)
print("- Čas výpočtu:", brute_force_time, "sekund")

print("\nŘešení simulovaným žíháním:")
print("- Nejlepší cena:", sa_price)
print("- Nejlepší řešení:", sa_solution)
print("- Čas výpočtu:", sa_time, "sekund")

# Vykreslení konvergenčního grafu pro simulované žíhání
plt.plot(best_prices)
plt.xlabel("Iterace")
plt.ylabel("Nejlepší cena")
plt.title("Konvergence simulovaného žíhání")
plt.show()
