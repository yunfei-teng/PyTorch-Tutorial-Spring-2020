# PyTorch tutorial codes for course Advanced Machine Learning
import torch
import torch.nn.functional as F

# 32-bit floating point
# floating-point arithmetic: https://en.wikipedia.org/wiki/Floating-point_arithmetic
dtype = torch.float32 
# put tensor on cpu(or you can try GPU)
device = torch.device("cpu")

# our data
POLY_DEGREE = 2
x = 2.0
x = [x** i for i in range(POLY_DEGREE + 1)]
w = torch.randn(POLY_DEGREE + 1, device=device, requires_grad = True)
x = torch.tensor(x, dtype=dtype, device=device, requires_grad = True)

target = 2.4
target = torch.tensor([target], dtype=dtype, device=device, requires_grad = True)

y = torch.sum(w* x) # forward
F.mse_loss(y, target.view_as(y)).backward() # backward

# (a) calculate the gradients manually
# y = w_0 + w_1 x + w_2 x^2
# grad(w) = x * (y - target) * 2
grad = x* (y - target)* 2
print('(a) The gradients of w are', grad.data.cpu().numpy())

# (b) double-check by PyTorch AutoGrad
print('(b) The gradients of w are', w.grad.data.cpu().numpy())

# simulate taking gradient descent steps
w.grad.zero_()
for i in range(10):
    y = torch.sum(w* x) # forward
    loss = F.mse_loss(y, target.view_as(y))
    loss.backward() # backward
    w.data.add_(-0.01, w.grad)
    w.grad.zero_()
    print('w is', w.data.cpu().numpy(), 'and y is', y.item())

