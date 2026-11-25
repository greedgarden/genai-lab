"""Fit y = sin(x) on [-pi, pi] with small MLPs.

Each experiment answers one question and sets its own seed, so it can be
reproduced on its own.
"""

import math
import torch
from torch import nn

N = 200     # number of data points

def make_data():
    """Return N points of y = sin(x) on [-pi, pi], both with shape (N, 1)."""
    x = torch.linspace(-math.pi, math.pi, N).unsqueeze(1)   # (N, 1)
    y = torch.sin(x)                                        # (N, 1), element-wise
    return x, y

def make_mlp(width, activation):
    """Return a model: Linear(1, width) -> activation -> Linear(width, 1)."""
    return nn.Sequential(
        nn.Linear(1, width),
        activation,
        nn.Linear(width, 1),
    )

def train_fn(model, x, y, lr, steps):
    """Train `model` with SGD on all the data for `steps` steps.

    The parameters of `model` are updated in place. Return the final MSE loss
    as a Python float.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()   # clear the gradients from the last step

        loss = loss_fn(model(x), y)
        loss.backward()         # compute the gradients

        optimizer.step()        # update the parameters: p -= lr * p.grad

    loss = loss_fn(model(x), y)
    return loss.item()

def count_params(model):
    """Return the number of trainable numbers in `model`."""
    return sum(p.numel() for p in model.parameters())

def exp1_activation(x, y):
    """Does a model need an activation function to fit a curve?

    A straight line cannot fit sin(x). Stacking two Linear layers does not help,
    because the result is still a straight line. Adding tanh between them does.
    """
    print('--- Experiment 1: activation function ---')
    torch.manual_seed(0)

    models = {}
    models['a'] = nn.Sequential(        # y = wx + b
        nn.Linear(1, 1),
    )
    models['b'] = nn.Sequential(        # two layers, no activation
        nn.Linear(1, 16),
        nn.Linear(16, 1),
    )
    models['c'] = nn.Sequential(        # two layers with tanh between them
        nn.Linear(1, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    )

    print(f'{"model":>6}{"loss":>14}{"params":>8}')
    for name, model in models.items():
        loss = train_fn(model, x, y, 0.1, 2000)
        print(f'{name:>6}{loss:>14.4g}{count_params(model):>8}')
    print()

def exp2_width(x, y):
    """Is a wider model always better?

    Compare tanh and ReLU with 1 to 64 neurons in the hidden layer.
    """
    print('--- Experiment 2: width (lr=0.1) ---')
    torch.manual_seed(0)

    models = {}
    for width in [1, 2, 4, 16, 64]:
        models[f'tanh{width}'] = make_mlp(width, nn.Tanh())
        models[f'relu{width}'] = make_mlp(width, nn.ReLU())

    print(f'{"model":>6}{"loss":>14}{"params":>8}')
    for name, model in models.items():
        # Keep lr=0.1 for every width on purpose: width 64 diverges to nan,
        # which experiment 3 explains.
        loss = train_fn(model, x, y, 0.1, 2000)
        print(f'{name:>6}{loss:>14.4g}{count_params(model):>8}')
    print()

def exp3_lr(x, y):
    """Why does width 64 diverge, and which learning rate works?

    With more neurons, more parameters move together in each step, so the same
    lr makes a bigger change to the output.
    """
    print('--- Experiment 3: learning rate (tanh64) ---')
    print(f'{"model":>6}{"loss":>14}{"lr":>8}')
    for lr in [0.1, 0.05, 0.002, 0.001]:
        # Create a new model for every lr, so a run that diverges to nan does
        # not affect the next one. Reset the seed first, so every model starts
        # from the same initial values and lr is the only difference.
        torch.manual_seed(0)
        model = make_mlp(64, nn.Tanh())
        loss = train_fn(model, x, y, lr, 2000)
        print(f'{"tanh64":>6}{loss:>14.4g}{lr:>8}')
    print()

def exp4_seeds(x, y):
    """Is the result reliable, or did we get lucky with the initial values?

    Train the same model with the same lr from different random initial values.
    """
    print('--- Experiment 4: random seed (tanh64, lr=0.05) ---')
    print(f'{"model":>6}{"loss":>14}{"seed":>8}')
    for seed in [10, 200, 63000]:
        torch.manual_seed(seed)     # a different seed gives different initial values
        model = make_mlp(64, nn.Tanh())
        loss = train_fn(model, x, y, 0.05, 2000)
        print(f'{"tanh64":>6}{loss:>14.4g}{seed:>8}')
    print()

def main():
    x, y = make_data()

    exp1_activation(x, y)
    exp2_width(x, y)
    exp3_lr(x, y)
    exp4_seeds(x, y)

if __name__ == "__main__":
    main()
