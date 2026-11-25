# 01-what-is-a-model

A model is a function with parameters. Training means changing the parameters so that a loss gets smaller, using the gradient of the loss.

This stage builds that idea in two parts:

1. Fit a straight line three times: with gradients computed by hand, with autograd, and with `nn.Linear` and an optimizer. All three give the same result.
2. Fit a curve, `sin(x)`, with small multi-layer perceptrons (MLPs), and run experiments on the activation function, width, learning rate, and random seed.

## Summary

- Training is gradient descent on a loss. Autograd computes the same gradient as the chain rule by hand.
- In PyTorch, gradients are added to `.grad`, so they must be cleared every step. Parameters must be updated in place, without recording the update.
- Shapes are part of correctness. Broadcasting can silently give the wrong result.
- Stacked linear layers are still linear. A non-linear activation function lets a model fit curves.
- A bigger model is not automatically better. It may need a different learning rate.
- Results depend on random initial values, so compare models with several seeds.

## Files

| File | What it does |
|---|---|
| `linear_manual.py` | Linear regression with a numeric gradient and a hand-derived gradient |
| `linear_autograd.py` | The same training with autograd, plus a gradient check against the formula |
| `linear_nn.py` | The same training with `nn.Linear`, `nn.MSELoss`, and `torch.optim.SGD` |
| `mlp_sin.py` | Four experiments that fit `sin(x)` with MLPs |

## Run

From the repo root, with the virtual environment `.venv` from [00-setup](../00-setup/README.md) active:

```bash
python 01-what-is-a-model/linear_manual.py
python 01-what-is-a-model/linear_autograd.py
python 01-what-is-a-model/linear_nn.py
python 01-what-is-a-model/mlp_sin.py
```

All scripts set a seed, so the output below is reproducible. The scripts run on the CPU.

## Part 1: Linear regression

### Data and loss

- 100 points with `x` evenly spaced on [-1, 1], and `y = 3x + 2 + noise`. The noise is normal with standard deviation 0.1.
- The model is `y = w*x + b`, starting from `w = 0` and `b = 0`.
- The loss is the mean squared error (MSE): `L = mean((w*x + b - y)^2)`.
- Each run trains for 100 steps with plain gradient descent: `w -= lr * dL/dw`, `b -= lr * dL/db`.
- `float64` is used, so the results can be compared to many digits.

### `linear_manual.py`: gradients by hand

Two ways to get the gradient.

**Numeric gradient.** Move a parameter by a small step $\epsilon$ (`eps` in the code), and divide the change in the loss by $\epsilon$:

$$
\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon}
$$

This is the slope of the straight line through two nearby points on the loss curve. It approximates the slope of the curve itself (the tangent) at `w`:

```
 L
 │ *
 │  *
 │   *
 │    ●  L(w)
 │     ╲   ← slope = (L(w + eps) - L(w)) / eps
 │      ●  L(w + eps)
 │        *
 │           *          *
 │               * * *
 └────┬─┬──────────────────── w
      w w+eps
```

The script uses `eps = 0.001`, and does the same for `b`.

**Formula gradient.** Derive the slope exactly with the chain rule. For point $i$, the error and the loss are

$$
e_i = w x_i + b - y_i
$$

$$
L = \frac{1}{N} \sum_{i=1}^{N} e_i^2
$$

$\partial L / \partial w$ is a partial derivative: it asks how $L$ changes when only $w$ changes, with $b$, $x$, and $y$ held constant. $L$ depends on $w$ only through each $e_i$, so the chain rule multiplies two simpler derivatives:

| Step | Derivative | Why |
|---|---|---|
| $L$ with respect to $e_i$ | $\dfrac{\partial L}{\partial e_i} = \dfrac{2 e_i}{N}$ | The derivative of $e^2$ is $2e$ |
| $e_i$ with respect to $w$ | $\dfrac{\partial e_i}{\partial w} = x_i$ | In $w x_i + b - y_i$, only $w x_i$ changes with $w$ |
| $e_i$ with respect to $b$ | $\dfrac{\partial e_i}{\partial b} = 1$ | In $w x_i + b - y_i$, only $b$ changes with $b$ |

Multiply and add over all points:

$$
\frac{\partial L}{\partial w}
= \sum_{i=1}^{N} \frac{\partial L}{\partial e_i} \cdot \frac{\partial e_i}{\partial w}
= \frac{1}{N} \sum_{i=1}^{N} 2 e_i x_i
$$

$$
\frac{\partial L}{\partial b}
= \sum_{i=1}^{N} \frac{\partial L}{\partial e_i} \cdot \frac{\partial e_i}{\partial b}
= \frac{1}{N} \sum_{i=1}^{N} 2 e_i
$$

A sum divided by $N$ is a mean, so `formula_grad` computes these as:

```python
e = w * x + b - y
dw = (2 * e * x).mean()
db = (2 * e).mean()
```

```
grad_fn           lr           w           b        loss
numeric_grad    0.01       1.504       1.735      0.8862
numeric_grad     0.1       3.038           2     0.01217
numeric_grad     1.0        3.04  -1.776e-11       4.014
numeric_grad     1.5   3.153e+12  -2.033e+12   7.514e+24
formula_grad    0.01       1.504       1.735      0.8857
formula_grad     0.1       3.038           2     0.01216
formula_grad     1.0       3.041   8.882e-16       4.014
formula_grad     1.5  -1.689e+13  -2.536e+30   6.431e+60
```

What the learning rate does:

- **lr = 0.01**: too small. After 100 steps, `w` and `b` are still far from the answer.
- **lr = 0.1**: converges to `w = 3.038`, `b = 2`.
- **lr = 1.0**: `w` converges, but `b` jumps between 0 and 4 on every step. For `b`, this lr is exactly twice the step that would land on the minimum, so each step goes past it by the same distance. After an even number of steps, `b` is back at about 0, and the loss is about `(0 - 2)^2 = 4`.
- **lr = 1.5**: diverges. Each step goes further past the minimum than the last one.

Other observations:

- The model fits the data, not the "true" line. The answer is `w = 3.038`, not 3, because the 100 noisy points are best fitted by that line.
- The numeric gradient is close to the formula, but not exact (0.8862 vs 0.8857 at lr = 0.01). When the loss becomes huge (lr = 1.5), it also loses precision: it subtracts two large, nearly equal numbers. So the numeric run diverges differently.
- At lr = 1.5, the value of `w` is not meaningful. The data is symmetric, so terms that should be exactly zero are tiny rounding errors (around 1e-18). Divergence amplifies them by many orders of magnitude.

### `linear_autograd.py`: autograd

With `requires_grad=True`, PyTorch records every operation into a graph. `loss.backward()` walks the graph backward with the chain rule and stores the gradients in `w.grad` and `b.grad`.

The gradient check compares autograd with the formula at two points:

```
    w0          b0   dw autograd    dw formula   db autograd    db formula  match
   0.0         0.0  -2.068263634  -2.068263634  -4.000883205  -4.000883205   True
   1.5        -0.5  -1.048061614  -1.048061614  -5.000883205  -5.000883205   True
```

Training gives the same result as the formula version:

```
    lr           w           b        loss
  0.01       1.504       1.735      0.8857
   0.1       3.038           2     0.01216
   1.0       3.041   -9.77e-15       4.014
   1.5   2.199e+12  -2.536e+30   6.431e+60
```

The table matches the formula version except in two places:

- **`b` at lr = 1.0**: `-9.77e-15` here and `8.882e-16` in the formula version. Both are 0 plus a rounding error.
- **`w` at lr = 1.5**: the run has diverged, so `w` is an amplified rounding error (see above).

The rounding errors differ because floating-point addition depends on the order. For example, in Python, `(0.1 + 0.2) + 0.3` is `0.6000000000000001`, but `0.1 + (0.2 + 0.3)` is `0.6`. Autograd and `formula_grad` compute the same math in a different order.

Two pitfalls when combining autograd with a hand-written update. They do not exist in `linear_manual.py`, because there the parameters do not use autograd and have no `.grad`:

1. **`w = w - lr * w.grad` creates a new tensor.** The name `w` then points to a tensor that is not the original leaf, so the next `backward()` does not fill its `.grad`. Inside `torch.no_grad()`, the new tensor does not require gradients at all, so the next `backward()` fails. The fix is to update in place inside `torch.no_grad()`: `w -= lr * w.grad`.
2. **`backward()` adds to `.grad`; it does not overwrite it.** Without `w.grad.zero_()` after each step, each step uses the sum of all previous gradients. With lr = 0.1, this gave `w = 1.793` instead of `3.038`.

Gradients add up on purpose: when a parameter is used more than once, its gradient is the sum over every path. For example, for `w*x + w`, `d/dw = x + 1`.

### `linear_nn.py`: `nn.Linear` and an optimizer

| By hand (`linear_autograd.py`) | PyTorch |
|---|---|
| `w`, `b`, and `w * x + b` | `nn.Linear(1, 1)`, which computes `x @ weight.T + bias` |
| `loss_fn` | `nn.MSELoss()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |
| `with torch.no_grad(): w -= lr * w.grad` | `optimizer.step()` |

The results match the formula version for lr = 0.01, 0.1, and 1.0.

Shapes matter:

- `nn.Linear` expects input with shape `(samples, features)`. The data has shape `(100,)`, so it is reshaped to `(100, 1)` with `x.unsqueeze(1)`.
- The target must have the same shape as the output. With output `(100, 1)` and target `(100,)`, broadcasting turns `output - target` into a `(100, 100)` tensor that compares every prediction with every target. PyTorch only prints a `UserWarning`. Training still runs, but it learns `w ≈ 0` and `b ≈ mean(y)` with a loss of 3.16.

## Part 2: Fitting `sin(x)` with MLPs

`mlp_sin.py` fits `y = sin(x)` with 200 points on [-pi, pi]. It uses `float32`, random initial values, plain SGD on all the data, and 2000 steps. Each experiment sets its own seed.

An MLP here is `Linear(1, width) -> activation -> Linear(width, 1)`. With width 16 and no activation:

```
 input        layer 1: nn.Linear(1, 16)              layer 2: nn.Linear(16, 1)       output
              16 neurons, each with its own w and b   1 neuron

            ┌─ h1  = w1*x  + b1  ─┐
            ├─ h2  = w2*x  + b2  ─┤
   x  ──────┼─ h3  = w3*x  + b3  ─┼──→  y = v1*h1 + v2*h2 + ... + v16*h16 + c  ──→  y
            │         ...         │
            └─ h16 = w16*x + b16 ─┘

 parameters:  w1..w16 (16) + b1..b16 (16) = 32         v1..v16 (16) + c (1) = 17
```

`nn.Linear(in, out)` has a weight of shape `(out, in)` and a bias of shape `(out,)`. So this model has 49 parameters.

### Experiment 1: Does a model need an activation function?

```
 model          loss  params
     a        0.1996       2      Linear(1, 1)
     b        0.1996      49      Linear(1, 16) -> Linear(16, 1)
     c     0.0006157      49      Linear(1, 16) -> Tanh -> Linear(16, 1)
```

- **a**: 0.1996 is the best a straight line can do.
- **b** has 49 parameters but gets exactly the same loss. Two linear layers are still a straight line:
  `y = sum(v_i * (w_i*x + b_i)) + c = (sum v_i*w_i) * x + (sum v_i*b_i + c)`.
- **c** adds `tanh` between the layers. Each term `v_i * tanh(w_i*x + b_i)` is an S-shaped curve. `w_i` sets how steep it is, `b_i` where it is, and `v_i` how tall it is and which way it faces. A sum of 16 such curves can bend into the shape of `sin(x)`.

More layers or more parameters do not help without a non-linear activation function.

### How width and an activation split `x` into pieces

Every neuron gets the same `x`. But each neuron has its own `b`, so each one starts to respond at a different `x`. The activation function is what makes this matter.

`ReLU` (`max(0, x)`) shows it most clearly. Take three neurons with `w = 1` and different `b`:

```
h1 = ReLU(x + 1)    is 0 for x < -1, then grows
h2 = ReLU(x)        is 0 for x <  0, then grows
h3 = ReLU(x - 1)    is 0 for x <  1, then grows

y  = 1*h1 - 2*h2 + 2*h3
```

```
   x    h1    h2    h3     y
  -2     0     0     0     0
  -1     0     0     0     0    <- h1 starts
-0.5   0.5     0     0   0.5
   0     1     0     0     1    <- h2 starts
 0.5   1.5   0.5     0   0.5
   1     2     1     0     0    <- h3 starts
 1.5   2.5   1.5   0.5   0.5
   2     3     2     1     1
```

```
 y
 1 │          ╱╲          ╱
   │        ╱    ╲      ╱
 0 │──────╱        ╲  ╱
   └──────┬────┬────┬──────── x
         -1    0    1
```

Each time `x` passes a neuron's starting point, that neuron joins the sum, and the slope of `y` changes:

| Range of `x` | Active neurons | Slope of `y` |
|---|---|---|
| x < -1 | none | 0 |
| -1 < x < 0 | h1 | 1 |
| 0 < x < 1 | h1, h2 | 1 - 2 = -1 |
| x > 1 | h1, h2, h3 | 1 - 2 + 2 = 1 |

Three neurons give three bends, which split the `x` axis into four straight pieces:

- The first layer (`w_i`, `b_i`) sets **where** each bend is: at `x = -b_i / w_i`.
- The second layer (`v_i`) sets **how much** the slope changes at each bend.

So the width is the number of bends the model can make. With width 16, training moves up to 16 bends and sets the slopes between them, to make a line with bends that follows `sin(x)`. Both parts are needed: without the activation, there are no bends (model b); with width 1, there is only one.

`tanh` works the same way, but each bend is a smooth S-shaped turn instead of a sharp corner. The center of each S-curve is also at `x = -b_i / w_i`.

### Experiment 2: Is a wider model always better?

`tanh` and `ReLU` (`max(0, x)`), with widths 1 to 64, all with lr = 0.1:

```
 model          loss  params
 tanh1       0.06541       4
 relu1        0.1629       4
 tanh2       0.01949       7
 relu2        0.1405       7
 tanh4      0.005207      13
 relu4      0.002458      13
tanh16     0.0007707      49
relu16     0.0002319      49
tanh64           nan     193
relu64           nan     193
```

- A width of `w` has `3w + 1` parameters.
- With `tanh`, the loss goes down steadily from width 1 to 16. One S-curve can only go up or down once, so width 1 cannot make the full wave.
- With `ReLU`, each neuron adds a bend, as shown above. The results are less regular. One cause is a **dead ReLU**: if a neuron's input is negative for every `x`, its output and gradient are always 0, so it never learns. In an earlier run, one of the two neurons of a width-2 ReLU model was dead, and the model did no better than width 1.
- At width 64, both diverge to `nan`.

### Experiment 3: Why does width 64 diverge?

`tanh` with width 64. A new model is created for each lr from the same seed, so lr is the only difference:

```
 model          loss      lr
tanh64           nan     0.1
tanh64     0.0008378    0.05
tanh64       0.02384   0.002
tanh64       0.05677   0.001
```

The output is a sum of 64 terms: `y = v1*h1 + ... + v64*h64 + c`. In each step, all 64 `v_i` move at the same time. Each step is small, but together they change `y` a lot. So the same lr is effectively larger for a wider model. With lr = 0.1, the loss went 1.0, 8.5, 76, 229, ..., and then `nan`.

- **lr too large (0.1)**: diverges.
- **lr too small (0.002, 0.001)**: does not get far in 2000 steps.
- **lr = 0.05** works for this model.

The learning rate has to fit the model. Values such as lr, steps, and width are **hyperparameters**: they are chosen by the person, not learned by training.

### Experiment 4: Is the result reliable?

The same model and lr, from three different random initial values:

```
 model          loss    seed
tanh64     0.0009474      10
tanh64      0.001224     200
tanh64     0.0007661   63000
```

With lr = 0.05, width 64 is reliable: the loss is between 0.0008 and 0.0012 for every seed. One run can be luck, so a comparison between models should use several seeds.

## Limitations

The loss here is measured on the training data only. It shows how well a model fits these 200 points, not how well it predicts new `x` values, for example outside [-pi, pi].
