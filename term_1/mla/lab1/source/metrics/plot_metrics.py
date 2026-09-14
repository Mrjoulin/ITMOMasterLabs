import matplotlib.pyplot as plt
import numpy as np
import os

IMAGES_DIR = "images/"


def _save_and_close(img_name: str):
    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path)
    plt.close('all')


def plot_hist(train_history, val_history=None, xlabel: str = "Batch", save_img: bool = True):
    plt.plot(train_history, c='b', label="Train")
    if val_history is not None:
        if len(val_history) < len(train_history):
            step = len(train_history) // len(val_history)
        else:
            val_history = val_history[:len(train_history)]
            step = 1
        plt.plot(
            range(0, len(train_history), step),
            val_history,
            c='r', label="Validation"
        )

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Loss')

    if save_img:
        _save_and_close(f"{xlabel}_history.png")


def plot_all_hist(train_history, val_history):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plot_hist(train_history, val_history, save_img=False)

    plt.subplot(1, 2, 2)
    step = len(train_history) // len(val_history)
    plot_hist(train_history[::step], val_history, xlabel="Epoch", save_img=False)

    _save_and_close("Train_history.png")


def plot_margins(model, X: np.ndarray, y: np.ndarray, noise_bound: float = -0.02, trusted_bound: float = 0.1):
    margins = model.forward(X) * y
    margins = np.sort(margins)

    plt.figure(figsize=(12, 5))

    noise_end = np.searchsorted(margins, noise_bound, side='right')
    trusted_start = np.searchsorted(margins, trusted_bound, side='right')

    noise_indx = np.arange(noise_end)
    borderline_indx = np.arange(noise_end, trusted_start)
    trusted_indx = np.arange(trusted_start, len(margins))

    # Plot figure

    # Areas
    plt.fill_between(noise_indx, margins[noise_indx], color='red', alpha=0.7)
    plt.fill_between(borderline_indx, margins[borderline_indx], color='yellow', alpha=0.7)
    plt.fill_between(trusted_indx, margins[trusted_indx], color='green', alpha=0.7)

    # Curve and x-line
    plt.plot(margins, color='blue')
    plt.axhline(0, color='black', linestyle='--')

    plt.xlabel('Samples')
    plt.ylabel('Margin')
    plt.title('Margins by samples')

    _save_and_close("Margins_by_samples.png")
