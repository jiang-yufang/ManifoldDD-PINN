import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import copy
from test_functions import detailed_evaluation, loss_curve
from basic_functions import *
from record_result import record_pinn_test_results

PRINT_ERROR_100TIME = True

def laplace(u,x):
    """
    Euclidean Laplace operator: sum of the second partial derivatives.

    Args:
        u: Function values at the points x.
        x: Input coordinates (batch of n-dimensional points).
    """
    n = x.shape[1]  # spatial dimension

    # First-order derivatives. `grad_outputs=torch.ones_like(u)` sums the
    # gradient over all sample points; `create_graph` keeps the graph so that
    # higher-order derivatives can be computed; `retain_graph` keeps the graph
    # for subsequent autograd calls; [0] unpacks the gradient tensor from the
    # returned tuple.
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True,
                                 retain_graph=True,
                                 )[0]

    # Second-order derivatives
    laplacian_term = torch.zeros_like(u)
    for i in range(n):
        # ∂²u/∂x_i² (the i:i+1 slice keeps the 2-D tensor structure)
        grad_u_i = grad_u[:, i:i+1]
        grad2_u_i = torch.autograd.grad(grad_u_i, x,
                                       grad_outputs=torch.ones_like(grad_u_i),
                                       create_graph=True, retain_graph=True)[0]
        laplacian_term += grad2_u_i[:, i:i+1]  # sum of the diagonal second derivatives
    return laplacian_term


def spherical_laplace(u, x):
    """
    ∆u = 4^(-1)(1+||x||^2)^2 * Σ(∂²u/∂x_j²) - 2^(-1)(n-2)(1+||x||^2) * Σ(x_j * ∂u/∂x_j)
    """
    n = x.shape[1]  # spatial dimension
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)

    # First-order derivatives (see laplace() for the autograd details)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True,
                                 retain_graph=True
                                 )[0]

    # Second-order derivatives
    laplacian_term1 = torch.zeros_like(u)
    for i in range(n):
        # ∂²u/∂x_i² (the i:i+1 slice keeps the 2-D tensor structure)
        grad_u_i = grad_u[:, i:i+1]
        grad2_u_i = torch.autograd.grad(grad_u_i, x,
                                       grad_outputs=torch.ones_like(grad_u_i),
                                       create_graph=True, retain_graph=True)[0]
        laplacian_term1 += grad2_u_i[:, i:i+1]
    # Σ(x_j * ∂u/∂x_j); keepdim preserves the 2-D shape of the result
    sum_x_grad = torch.sum(x * grad_u, dim=1, keepdim=True)

    # Spherical Laplace operator
    delta_u = (1/4) * (1 + norm_x_squared)**2 * laplacian_term1 - \
              (1/2) * (n - 2) * (1 + norm_x_squared) * sum_x_grad

    return delta_u

class ReLU2(nn.Module):
    def forward(self, x):
        return torch.relu(x)**2

def f_function(x, n, b):
    """
    Source term f = (n + b)(1-||x||^2)/(1+||x||^2)
    """
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)
    return (n + b) * (1 - norm_x_squared) / (1 + norm_x_squared)

def exact_solution(x):
    """
    Exact solution u = (1-||x||^2)/(1+||x||^2)
    """
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)
    return (1-norm_x_squared) / (1+norm_x_squared)

def boundary_condition(x, R):
    """
    Boundary condition u = (1-R^2)/(1+R^2)

    Args:
        x: Input coordinates.
        R: Radius of the subdomain.
    """
    return (1 - R**2) / (1 + R**2)

class Train_pinn1:
    def __init__(self, n=2, b=1.0, R=1.0, hidden_dim=50, layers=4, epochs=1000, points=1000, md='PINN', device=None, dtype=torch.float32,
               act_function=nn.Tanh,interior_points_rate=0.8, interior_weight=0.5, opt='Adam', lr=0.001, lrs='exp', gamma=0.999, 
               f_function=f_function, boundary_condition=boundary_condition, exact_solution=exact_solution):
        """Initialize the training task."""
        self.n = n
        self.b = b
        self.R = R
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.epochs = epochs
        self.points = points
        self.md = md
        self.device = device
        self.dtype = dtype
        self.act_function = act_function
        self.interior_points_rate = interior_points_rate
        self.interior_weight = interior_weight
        self.opt = opt
        self.lr = lr
        self.lrs = lrs
        self.gamma = gamma
        self.f_function = f_function
        self.boundary_condition = boundary_condition
        self.exact_solution = exact_solution
        self.loss_history = None
        self.error_history = None
        self.training_time = None
        self.detailed_results = None
        if torch.cuda.is_available():
            # Initialize the CUDA context explicitly to avoid warnings
            torch.cuda.empty_cache()

        if isinstance(md, nn.Module):
            self.model = md
        elif md == 'PINN':
            self.model = PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)
        elif md == 'Res':
            self.model = Res_PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)

        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.lrs == 'exp':
            # Exponential decay: lr = lr * gamma after each epoch
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    def pinnfunction(self, model, x):
        """Raw neural network output (no post-processing)."""
        return model(x)

    def get_interior_points(self, N=None):
        if N is None:
            N=round(self.points*self.interior_points_rate)
        return UniformBall(N, self.n, self.device, self.dtype) * self.R
    def get_boundary_points(self, N=None):
        if N is None:
            N=round(self.points*(1 - self.interior_points_rate))
        return UniformSphericalSurface(N, self.n-1, self.device, self.dtype) * self.R

    def interior_loss(self, model, pinnfunction, x_interior):
        """
        Interior (PDE residual) loss: -∆u + b·u - f = 0
        """
        x_interior.requires_grad_(True)
        u_pred = pinnfunction(model, x_interior)

        # Laplace operator
        delta_u = spherical_laplace(u_pred, x_interior)

        # Equation residual
        f_val = self.f_function(x_interior, self.n, self.b)
        residual = -delta_u + self.b * u_pred - f_val

        return torch.mean(residual**2)

    def boundary_loss(self, model, pinnfunction, x_boundary):
        """
        Boundary loss: u - u_boundary = 0
        """
        u_pred = pinnfunction(model, x_boundary)
        u_boundary = self.boundary_condition(x_boundary, self.R)
        return torch.mean((u_pred - u_boundary)**2)



    def get_iter_points(self, N=100000):
        return UniformSphericalSurface(N, self.n-1, self.device, self.dtype) / self.R

    def train(self):
        start_time = time.time()
        model, loss_history, error_history = train_progress(self.model, self.pinnfunction, self.get_interior_points, self.get_boundary_points, self.epochs, self.interior_loss, self.boundary_loss, self.interior_weight, self.optimizer, self.scheduler, self.exact_solution)
        end_time = time.time()
        self.model = model
        self.loss_history = loss_history
        self.error_history = error_history
        self.training_time = end_time - start_time
        loss_curve(loss_history, error_history)

    def test(self):
        detailed_results = detailed_evaluation(self.model, self.pinnfunction, self.get_interior_points, self.get_boundary_points, self.get_iter_points, self.exact_solution)
        self.detailed_results = detailed_results


def train_progress(model, pinnfunction, get_interior_points, get_boundary_points, epochs, interior_loss, boundary_loss, interior_weight, optimizer, scheduler, exact_solution):
    loss_history = []
    error_history = []

    best_model_state = None
    best_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        x_interior = get_interior_points()
        x_boundary = get_boundary_points()
        loss_interior = interior_loss(model, pinnfunction, x_interior)
        loss_boundary = boundary_loss(model, pinnfunction, x_boundary)
        total_loss = interior_weight*loss_interior + (1-interior_weight)*loss_boundary

        loss_history.append(total_loss.item())  # .item() returns the value of a single-element tensor
        with torch.no_grad():
            x_interior_ = x_interior.detach().clone()
            u_pred = pinnfunction(model, x_interior_)
            u_exact = exact_solution(x_interior_)
            error = torch.sqrt(torch.mean((u_exact - u_pred)**2))
            error_history.append(error.item())

        if epoch % 100 == 0 and PRINT_ERROR_100TIME:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                f"Interior Loss: {loss_interior.item():.6f}, "
                f"Boundary Loss: {loss_boundary.item():.6f}, "
                f"exact error: {error.item():.6f}")

        # Keep the best model state seen so far
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        optimizer.zero_grad()  # clear the gradients
        total_loss.backward()  # backpropagation
        optimizer.step()       # parameter update

        scheduler.step()

    x_interior = get_interior_points()
    x_boundary = get_boundary_points()
    loss_interior = interior_loss(model, pinnfunction, x_interior)
    loss_boundary = boundary_loss(model, pinnfunction, x_boundary)
    total_loss = interior_weight*loss_interior + (1-interior_weight)*loss_boundary

    loss_history.append(total_loss.item())
    with torch.no_grad():
        x_interior_ = x_interior.detach().clone()
        u_pred = pinnfunction(model, x_interior_)
        u_exact = exact_solution(x_interior_)
        error = torch.sqrt(torch.mean((u_exact - u_pred)**2))
        error_history.append(error.item())

    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch

    # Restore the best model state found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model found at epoch {best_epoch}, lowest loss {best_loss:.6f}")

    return model, loss_history, error_history

if __name__=="__main__":

    n = 2
    b = 1.0
    R = 1.2
    hidden_dim = 50
    layers = 2
    epochs = 1000
    points = 1000
    md = 'PINN'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    dtype = torch.float32
    act_function = nn.Tanh
    interior_points_rate = 0.5
    interior_weight = 0.5
    opt = 'Adam'
    lr = 0.001
    lrs = 'exp'
    gamma = 0.999

    task = Train_pinn1(n, b, R, hidden_dim, layers, epochs, points, md, device, dtype,
               act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma, 
               f_function, boundary_condition, exact_solution)
    task.train()
    task.test()
    record_pinn_test_results(n, b, R, hidden_dim, layers, epochs, points, md, device, dtype,
               act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma, 
               task.training_time, task.detailed_results['relative_l2'], 'pinn_test_results')
