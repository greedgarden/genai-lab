import torch

N = 100
NOISE_STD = 0.1
TRUE_W, TRUE_B = 3.0, 2.0
STEPS = 100
LOG_EVERY = 10
IS_SHOW_TRAINING_LOG = False

def make_data():
    x = torch.linspace(-1, 1, N, dtype=torch.float64)
    noise = NOISE_STD * torch.randn(N, dtype=torch.float64)
    y = TRUE_W * x + TRUE_B + noise
    return x, y

def loss_fn(w, b, x, y):
    """Return the mean squared error (MSE) of the line y = w*x + b."""
    y_pred = w * x + b
    return ((y_pred - y) ** 2).mean()

def numeric_grad(w, b, x, y):
    """Return approximate gradients (dL/dw, dL/db) by finite differences.

    Each parameter is moved by a small step eps, and the change in the loss is
    divided by eps. This approximates the slope, but it loses precision when the
    loss is large, because it subtracts two nearly equal numbers.
    """
    eps = 0.001
    base = loss_fn(w, b, x, y)
    dw = (loss_fn(w + eps, b, x, y) - base) / eps
    db = (loss_fn(w, b + eps, x, y) - base) / eps
    return dw, db

def formula_grad(w, b, x, y):
    """Return the exact gradients (dL/dw, dL/db) of the MSE loss.

    For each point, the error is e = w*x + b - y, and the loss is L = mean(e^2).
    By the chain rule:
        dL/dw = mean(dL/de * de/dw) = mean(2e * x)
        dL/db = mean(dL/de * de/db) = mean(2e * 1) = mean(2e)
    """
    e = w * x + b - y
    dw = (2 * e * x).mean()
    db = (2 * e).mean()
    return dw, db

def train_fn(x, y, lr, grad_fn):
    w, b = 0.0, 0.0
    if IS_SHOW_TRAINING_LOG:
        print(f'--- {grad_fn.__name__} with w={w}, b={b}, learning rate={lr} ---')
    for step in range(STEPS):
        loss = loss_fn(w, b, x, y)
        dw, db = grad_fn(w, b, x, y)

        w = w - lr * dw
        b = b - lr * db

        if IS_SHOW_TRAINING_LOG and step % LOG_EVERY == 0:
            print(f'loss={loss.item():.2f}, w={w.item():.2f}, dw={dw:.2f}, b={b.item():.2f}, db={db:.2f} @{step}')
    if IS_SHOW_TRAINING_LOG:
        print()
    # Compute the loss of the final w and b. The loss inside the loop was
    # computed before the last update, so it does not match the returned w and b.
    loss = loss_fn(w, b, x, y)
    return w, b, loss

def main():
    torch.manual_seed(0)
    x, y = make_data()

    gradients = [numeric_grad, formula_grad]
    learning_rates = [0.01, 0.1, 1.0, 1.5]
    summary = []    # one row per run: (grad_fn name, lr, w, b, loss)
    for grad_fn in gradients:
        for lr in learning_rates:
            w, b, loss = train_fn(x, y, lr, grad_fn)
            summary.append((grad_fn.__name__, lr, w.item(), b.item(), loss.item()))

    print('--- Summary ---')
    print(f'{"grad_fn":<14}{"lr":>6}{"w":>12}{"b":>12}{"loss":>12}')
    for name, lr, w, b, loss in summary:
        print(f'{name:<14}{lr:>6}{w:>12.4g}{b:>12.4g}{loss:>12.4g}')

if __name__ == "__main__":
    main()
