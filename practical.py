import numpy as np

# Config
num_classes = 5
num_items_per_class = 3


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


# Run
weights, prices = generate_mckp_instance(num_classes, num_items_per_class)
print(weights)
print(prices)
