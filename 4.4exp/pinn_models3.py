import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import copy
from test_functions import detailed_evaluation, loss_curve
from basic_functions import *
from record_result import record_pinn_test_results

PRINT_ERROR_100TIME = True

def get_w_norm2(x):
    """Squared moduli |w_j|^2 of each complex coordinate of the chart point x."""
    n = x.shape[1]  # spatial dimension
    N = x.shape[0]
    n_ = round(n/2)
    w_norm2 = torch.zeros(N,n_).to(device=x.device,dtype=x.dtype)
    for i in range(n_):
        w_norm2[:,i:i+1] = x[:,i:i+1]**2 + x[:,n_+i:n_+i+1]**2
    return w_norm2

def cp_laplace(u, x):
    """
    ∆u = (1 + ||x||^2) * ( Σ ∂²u/∂x_i²  +  second-order cross terms )
    For the full expression, see Section 4.4 of the paper.
    """
    n = x.shape[1]  # spatial dimension
    n_ = round(n/2)
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)

    # First-order derivatives (see laplace() in pinn_models1 for the autograd details)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True,
                                 retain_graph=True
                                 )[0]

    # Second-order derivatives: diagonal terms plus the cross terms of the
    # complex structure
    laplacian_term = torch.zeros_like(u)
    second_derivative_term1 = torch.zeros_like(u)
    second_derivative_term2 = torch.zeros_like(u)
    for i in range(n_):
        grad_u_x_i = grad_u[:, i:i+1]
        grad_u_y_i = grad_u[:, n_+i:n_+i+1]
        grad2_u_x_i = torch.autograd.grad(grad_u_x_i, x,
                                       grad_outputs=torch.ones_like(grad_u_x_i),
                                       create_graph=True, retain_graph=True)[0]
        grad2_u_y_i = torch.autograd.grad(grad_u_y_i, x,
                                       grad_outputs=torch.ones_like(grad_u_y_i),
                                       create_graph=True, retain_graph=True)[0]
        laplacian_term = laplacian_term + grad2_u_x_i[:, i:i+1] + grad2_u_y_i[:, n_+i:n_+i+1]
        second_derivative_term1 = second_derivative_term1 + torch.sum((x[:,i:i+1] * x[:,0:n_] + x[:,n_+i:n_+i+1] * x[:,n_:]) * (grad2_u_x_i[:, 0:n_] + grad2_u_y_i[:, n_:]), dim=1, keepdim=True)
        second_derivative_term2 = second_derivative_term2 + torch.sum((x[:,i:i+1] * x[:,n_:] - x[:,n_+i:n_+i+1] * x[:,0:n_]) * (grad2_u_x_i[:, n_:] - grad2_u_y_i[:, 0:n_]), dim=1, keepdim=True)

    # CP^k Laplace operator
    delta_u = (1 + norm_x_squared) * (laplacian_term + second_derivative_term1 + second_derivative_term2)

    return delta_u

class ReLU2(nn.Module):
    def forward(self, x):
        return torch.relu(x)**2

def f_function(x, n, b, a):
    """
    Source term f = (4n + 4 + b)*u - 4Σa_i
    """
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)
    w_norm2 = get_w_norm2(x)
    return (4*n + b + 4) * (a[0] + torch.sum(a[1:] * w_norm2, dim=1, keepdim=True)) / (1 + norm_x_squared) - 4 * torch.sum(a)

def exact_solution(x,a):
    """
    Exact solution u = (ai + Σaj*|wj|^2) / (1 + ||w||^2)
    """
    norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)
    w_norm2 = get_w_norm2(x)
    return (a[0] + torch.sum(a[1:] * w_norm2, dim=1, keepdim=True)) / (1 + norm_x_squared)

def boundary_condition(x, R, a):
    """
    Boundary condition u = (ai + R^2 * Σaj) / (1 + n * R^2)
    """
    return(exact_solution(x,a))

class Train_pinn3:
    def __init__(self, n=2, b=1.0, a=torch.tensor([1,-1]), R=1.0, hidden_dim=50, layers=4, epochs=1000, points=1000, md='PINN', device=None, dtype=torch.float32,
               act_function=nn.Tanh,interior_points_rate=0.8, interior_weight=0.5, opt='Adam', lr=0.001, lrs='exp', gamma=0.999, 
               f_function=f_function, boundary_condition=boundary_condition, exact_solution=exact_solution):
        """Initialize the training task."""
        self.n = n
        self.b = b
        self.a = a
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
        # Wrap the exact solution so that only x needs to be passed
        # (the coefficient vector a is fixed for this subdomain).
        n_ = n*2
        def exact_solution_a(x):
            return exact_solution(x,self.a)

        self.exact_solution = exact_solution_a
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
            self.model = PINN(input_dim=n_, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)
        elif md == 'Res':
            self.model = Res_PINN(input_dim=n_, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)

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
        # Uniform sampling in the unit ball via polar coordinates: the first
        # n coordinates are the real parts, the last n the imaginary parts.
        angles = torch.rand(N, self.n) * 2 * torch.pi
        angles = angles.to(device=self.device, dtype=self.dtype)
        radius = torch.sqrt(torch.rand(N, self.n)).to(device=self.device, dtype=self.dtype)
        return torch.cat([torch.cos(angles) * radius, torch.sin(angles) * radius], dim=1) * self.R

    def get_boundary_points(self, N=None):
        if N is None:
            N=round(self.points*(1 - self.interior_points_rate))
        # Boundary of the ball: radius 1 in exactly one complex coordinate
        N_ = round(N/self.n)
        angles = torch.rand(self.n * N_, self.n) * 2 * torch.pi
        angles = angles.to(device=self.device, dtype=self.dtype)
        radius_ = torch.sqrt(torch.rand(self.n * N_, self.n-1)).to(device=self.device, dtype=self.dtype)
        radius = torch.zeros(self.n * N_, self.n).to(device=self.device, dtype=self.dtype)
        row_indices = torch.arange(self.n * N_).to(device=self.device, dtype=self.dtype)
        k_indices = (row_indices // N_).unsqueeze(1)
        col_indices = torch.arange(self.n).unsqueeze(0).expand(self.n * N_, self.n).to(device=self.device, dtype=self.dtype)
        mask = col_indices == k_indices
        radius[mask] = 1.
        radius[~mask] = radius_.flatten()
        return torch.cat([torch.cos(angles) * radius, torch.sin(angles) * radius], dim=1) * self.R

    def interior_loss(self, model, pinnfunction, x_interior):
        """
        Interior (PDE residual) loss: -∆u + b·u - f = 0
        """
        x_interior.requires_grad_(True)
        u_pred = pinnfunction(model, x_interior)

        # Laplace operator
        delta_u = cp_laplace(u_pred, x_interior)

        # Equation residual
        f_val = self.f_function(x_interior, self.n, self.b, self.a)
        residual = -delta_u + self.b * u_pred - f_val

        return torch.mean(residual**2)

    def boundary_loss(self, model, pinnfunction, x_boundary):
        """
        Boundary loss: u - u_boundary = 0
        """
        u_pred = pinnfunction(model, x_boundary)
        u_boundary = self.boundary_condition(x_boundary, self.R, self.a)
        return torch.mean((u_pred - u_boundary)**2)



    def train(self):
        # Reset the learning rate before each training run.
        self.optimizer.param_groups[0]['lr'] = self.lr
        start_time = time.time()
        model, loss_history, error_history = train_progress(self.model, self.pinnfunction, self.get_interior_points, self.get_boundary_points, self.epochs, self.interior_loss, self.boundary_loss, self.interior_weight, self.optimizer, self.scheduler, self.exact_solution)
        end_time = time.time()
        self.model = model
        self.loss_history = loss_history
        self.error_history = error_history
        self.training_time = end_time - start_time
        loss_curve(loss_history, error_history)

    def test(self):
        detailed_results = detailed_evaluation(self.model, self.pinnfunction, self.get_interior_points, self.get_boundary_points, self.exact_solution)
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
            print(f"Epoch {epoch}, learning rate: {optimizer.param_groups[0]['lr']}, Total Loss: {total_loss.item():.6f}, "
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

    loss_history.append(total_loss.item())  # .item() returns the value of a single-element tensor
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

    n = 3
    b = 1.0
    a = torch.tensor([1.,2.,-1.,-2.])
    R = 1.2
    hidden_dim = 500
    layers = 3
    epochs = 5000
    points = 2000
    md = 'Res'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    dtype = torch.float32
    a = a.to(device=device, dtype=dtype)
    act_function = nn.Tanh
    interior_points_rate = 0.4
    interior_weight = 0.004
    opt = 'Adam'
    lr = 0.001
    lrs = 'exp'
    gamma = 0.999

    task = Train_pinn3(n, b, a, R, hidden_dim, layers, epochs, points, md, device, dtype,
               act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma, 
               f_function, boundary_condition, exact_solution)
    task.train()
    task.test()
    record_pinn_test_results(n, b, a, R, hidden_dim, layers, epochs, points, md, device, dtype,
               act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma, 
               task.training_time, task.detailed_results['relative_l2'], 'pinn3_test_results')
