import torch
from linear_manual import STEPS, make_data, loss_fn, formula_grad


def train_fn(x, y, lr):
    w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for step in range(STEPS):
        loss = loss_fn(w, b, x, y)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    loss = loss_fn(w, b, x, y)
    return w, b, loss

def check_grad(x, y, w0, b0):
    """Compare the autograd gradients with formula_grad at the point (w0, b0)."""
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)

    loss = loss_fn(w, b, x, y)
    loss.backward()

    dw_formula, db_formula = formula_grad(w0, b0, x, y)

    match = torch.allclose(w.grad, dw_formula) and torch.allclose(b.grad, db_formula)
    print(f'{w0:>6}{b0:>12}{w.grad:>14.10g}{dw_formula:>14.10g}'
            f'{b.grad:>14.10g}{db_formula:>14.10g}{match!s:>7}')


def main():
    torch.manual_seed(0)
    x, y = make_data()

    print('--- Gradient check: autograd vs formula_grad ---')
    print(f'{"w0":>6}{"b0":>12}{"dw autograd":>14}{"dw formula":>14}{"db autograd":>14}{"db formula":>14}{"match":>7}')
    check_grad(x, y, 0.0, 0.0)
    check_grad(x, y, 1.5, -0.5)

    learning_rates = [0.01, 0.1, 1.0, 1.5]
    summary = []    # one row per run: (lr, w, b, loss)
    for lr in learning_rates:
        w, b, loss = train_fn(x, y, lr)
        summary.append((lr, w.item(), b.item(), loss.item()))

    print('--- Summary ---')
    print(f'{"lr":>6}{"w":>12}{"b":>12}{"loss":>12}')
    for lr, w, b, loss in summary:
        print(f'{lr:>6}{w:>12.4g}{b:>12.4g}{loss:>12.4g}')


if __name__ == "__main__":
    main()