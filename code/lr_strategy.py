# Learning reate Linear Annealing strategy similar to one in the original CycleGAN paper


import click
import numpy as np
import matplotlib.pyplot as plt


@click.command()
@click.option("--initial_lr", default=2e-4, help="Initial learning rate")
@click.option(
    "--constant_epochs",
    default=100,
    help="Number of epochs with constant learning rate",
)
@click.option(
    "--anneal_epochs", default=50, help="Number of epochs for linear annealing"
)
def visualize_lr_schedule(initial_lr, constant_epochs, anneal_epochs):
    def linear_annealing(initial_lr, final_lr, anneal_epochs):
        def schedule(epoch):
            if epoch < anneal_epochs:
                return initial_lr
            else:
                return initial_lr + (final_lr - initial_lr) * (
                    (epoch - anneal_epochs) / anneal_epochs
                )

        return schedule

    # Generate the epochs
    epochs_constant = np.arange(0, constant_epochs)
    epochs_anneal = np.arange(constant_epochs, constant_epochs + anneal_epochs)

    # Generate the learning rates using the linear annealing function
    lr_schedule = linear_annealing(initial_lr, 0, anneal_epochs)
    learning_rates = [lr_schedule(epoch) for epoch in epochs_constant] + [
        lr_schedule(epoch) for epoch in epochs_anneal
    ]

    # Plot the learning rate schedule
    plt.plot(
        np.arange(1, constant_epochs + anneal_epochs + 1),
        learning_rates,
        label="Linear Annealing",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Linear Learning Rate Annealing")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    visualize_lr_schedule()
