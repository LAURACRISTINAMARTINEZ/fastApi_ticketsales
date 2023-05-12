import matplotlib.pyplot as plt
import numpy as np


def plot_training(hist: dict, save_path: str = None):

    epochs = len(hist.history['mse'])
    best_epoch = np.argmin(hist.history['val_loss'])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(range(epochs), hist.history['mse'], label='Train', marker='8', color="#4836d1")
    axs[0].plot(range(epochs), hist.history['val_mse'], label='Validacion', marker='8', color="#ffad42")
    axs[0].axvline(best_epoch, linestyle='--', color="#878787")
    axs[0].set_title('mse')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(range(epochs), hist.history['loss'], label='Train', marker='8', color="#4836d1")
    axs[1].plot(range(epochs), hist.history['val_loss'], label='Validacion', marker='8', color="#ffad42")
    axs[1].axvline(best_epoch, linestyle='--', color="#878787")
    axs[1].set_title('Loss')
    axs[1].grid(True)
    axs[1].legend()
    plt.xlabel('EPOCHS')


    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_val(x_val: np.ndarray, y_val: np.ndarray, model, save_path:str):
    results= model.predict(x_val)
    plt.scatter(range(len(y_val)),y_val,c='g')
    plt.scatter(range(len(results)),results,c='r')
    plt.title('Validate Test')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

