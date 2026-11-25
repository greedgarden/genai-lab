import math
import torch
from torch import nn

N = 200
STEPS = 2000
LR = 0.05

def make_data():
    x = torch.linspace(-math.pi, math.pi, N).unsqueeze(1)   # (N, 1)
    y = torch.sin(x)
    return x, y

def make_mlp(width, activation):
    """Return a model: Linear(1, width) -> activation -> Linear(width, 1)."""
    return nn.Sequential(
        nn.Linear(1, width),
        activation,
        nn.Linear(width, 1),
    )

def train_fn(model, x, y):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for step in range(STEPS):
        optimizer.zero_grad()

        loss = loss_fn(model(x), y)
        loss.backward()

        optimizer.step()

    loss = loss_fn(model(x), y)
    return loss.item()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    torch.manual_seed(0)
    x, y = make_data()

    models = {}
    models['a'] = nn.Sequential(
        nn.Linear(1, 1),
    )

    models['b'] = nn.Sequential(
        nn.Linear(1, 16),
        nn.Linear(16, 1),
    )

    models['c'] = nn.Sequential(
        nn.Linear(1, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    )

    for width in [1, 2, 4, 16, 64]:
        models[f'tanh{width}'] = make_mlp(width, nn.Tanh())
        models[f'relu{width}'] = make_mlp(width, nn.ReLU())

    print(f'{"model":>6}{"loss":>14}{"params":>8}')
    for name, model in models.items():
        loss = train_fn(model, x, y)
        print(f'{name:>6}{loss:>14.4g}{count_params(model):>8}')


if __name__ == "__main__":
    main()