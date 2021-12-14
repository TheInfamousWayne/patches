import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd


def plot_training_loss(task_name):
    path = Path(f'logs/{task_name}')

    train_log = pd.read_csv(str(path/"train.csv"), header=None)
    test_log = pd.read_csv(str(path/"test.csv"), header=None)

    # train curve
    plt.clf()
    plt.cla()
    x = train_log[0]
    y = train_log[1]
    plt.plot(x, y)
    plt.xlabel("epochs")
    plt.ylabel("MSE loss")
    plt.title(f'{task_name} training loss')

    save_path = Path(f'save/{task_name}')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}/train.png")

    # test curve
    plt.clf()
    plt.cla()
    x = test_log[0]
    y = test_log[1]
    plt.plot(x, y)
    plt.xlabel("epochs")
    plt.ylabel("MSE loss")
    plt.title(f'{task_name} testing loss')

    save_path = Path(f'save/{task_name}')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}/test.png")


if __name__ == '__main__':
    tasks = ['m_alpha_aaa', 'm_glu', 'm_h1', 'm_orn', 'm_putrescine', 'm_taurine']
    for task_name in tasks:
        plot_training_loss(task_name)