import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def select_samples(xs: np.ndarray, ys: np.ndarray, num: int):
    buckets = {i: [] for i in range(10)}
    class_counts = {i: 0 for i in range(10)}

    for x, y in zip(xs, ys):
        if class_counts[y] < num:
            buckets[y].append(x)
            class_counts[y] += 1

            if all(count == num for count in class_counts.values()):
                break

    selected_xs = [x for digit in range(10) for x in buckets[digit]]
    selected_ys = [digit for digit in range(10) for _ in buckets[digit]]

    return selected_xs, selected_ys


def visualize_mnist_1d(xs: np.ndarray, ys: np.ndarray, t: np.ndarray, num: int = 3):
    xs, ys = select_samples(xs, ys, num=num)

    rows, cols = 3, 9
    ratio = 2.6
    fig = plt.figure(figsize=[cols * 1.5, rows * 1.5 * ratio], dpi=60)

    for r in range(rows):
        for c in range(cols):
            ix = c * rows + r
            x = xs[ix]
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)

            ax.plot(x, t, 'k-', linewidth=2)
            ax.set_title(str(ys[ix]), fontsize=22)

            ax.invert_yaxis()
            ax.set_xticks([]), ax.set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()


def plot_training_stats(train_losses: list[float], val_accuracies: list[float], figsize: tuple[int, int] = (10, 4)):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=figsize)

    # Training loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Validation accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_test_dataset(x_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray, path: str, num_samples: int):
    x_test = x_test[:num_samples]
    y_test = y_test[:num_samples]
    predictions = predictions[:num_samples]

    with open(path, 'w') as f:
        for i in range(x_test.shape[0]):
            label = y_test[i]
            pred = predictions[i]
            x = x_test[i]

            line = f"{label} {pred} {' '.join(map(str, x))}\n"
            f.write(line)
