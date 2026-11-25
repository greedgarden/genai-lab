import torch
from torch import nn
from linear_manual import STEPS, make_data

def train_fn(x, y, lr):
    model = nn.Linear(1, 1, dtype=torch.float64)

    # nn.Linear starts with random values. Start from w = 0 and b = 0 instead,
    # so the results can be compared with linear_manual.py.
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # update rule: p -= lr * p.grad

    # nn.Linear expects the shape (samples, features).
    # (N,) -> (N, 1): N samples, 1 feature each
    X = x.unsqueeze(1)

    # Y must have the same shape as model(X). With shape (N,), the loss would
    # broadcast (N, 1) - (N,) to (N, N) and compare every prediction with every target.
    Y = y.unsqueeze(1)

    for step in range(STEPS):
        optimizer.zero_grad()   # clear the gradients from the last step

        output = model(X)       # forward pass: output = X @ weight.T + bias
        loss = loss_fn(output, Y)

        loss.backward()         # backward pass: compute the gradients
        optimizer.step()        # update the parameters with the gradients

    loss = loss_fn(model(X), Y)
    return model.weight.item(), model.bias.item(), loss.item()

def main():
    torch.manual_seed(0)
    x, y = make_data()

    learning_rates = [0.01, 0.1, 1.0, 1.5]
    summary = []    # one row per run: (lr, w, b, loss)
    for lr in learning_rates:
        w, b, loss = train_fn(x, y, lr)
        summary.append((lr, w, b, loss))

    print('--- Summary ---')
    print(f'{"lr":>6}{"w":>12}{"b":>12}{"loss":>12}')
    for lr, w, b, loss in summary:
        print(f'{lr:>6}{w:>12.4g}{b:>12.4g}{loss:>12.4g}')


if __name__ == "__main__":
    main()