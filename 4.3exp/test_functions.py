import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
def detailed_evaluation(model, pinnfunction, get_interior_points, get_boundary_points, get_iter_points, exact_solution):
    """
    Evaluate the accuracy of a PINN model in detail.

    Computes the relative L2 error on a large set of interior points, plus the
    relative L2 errors on the physical boundary and on the iteration (interface)
    boundary, and saves two diagnostic plots.
    """
    # Generate a large set of test points (the origin is added explicitly)
    x_test = get_interior_points(100000)
    x_test = torch.cat((x_test, torch.zeros(1,x_test.shape[1]).to(device=x_test.device, dtype=x_test.dtype)), dim=0)
    # Predictions
    with torch.no_grad():
        u_pred = pinnfunction(model, x_test)

    # Exact values
    u_exact = exact_solution(x_test)

    # Make sure both tensors live on the same device
    if u_pred.device != u_exact.device:
        u_exact = u_exact.to(u_pred.device)

    # Error metrics
    mse = torch.mean((u_pred - u_exact)**2)
    rmse = torch.sqrt(mse)
    relative_l2 = rmse / torch.sqrt(torch.mean(u_exact**2))

    print(f"Relative L2 error: {relative_l2.item()*100:.6f}%")

    # Boundary condition check (the two boundary parts are stacked by
    # get_boundary_points; concatenate them into one set of points)
    x_boundary = get_boundary_points(100000)
    x_boundary = torch.cat((x_boundary[0],x_boundary[1]), dim=0)
    with torch.no_grad():
        u_boundary_pred = pinnfunction(model, x_boundary)
    u_boundary_exact = exact_solution(x_boundary)

    boundary_error = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_exact)**2))/ torch.sqrt(torch.mean(u_boundary_exact**2))
    print(f"Boundary condition relative L2 error: {boundary_error.item()*100:.6f}%")

    # Iteration (interface) boundary check
    x_iter_boundary = get_iter_points(100000)
    with torch.no_grad():
        u_iter_pred = pinnfunction(model, x_iter_boundary)
    u_iter_exact = exact_solution(x_iter_boundary)
    iter_error = torch.sqrt(torch.mean((u_iter_pred - u_iter_exact)**2))/ torch.sqrt(torch.mean(u_iter_exact**2))
    print(f"Iteration boundary relative L2 error: {iter_error.item()*100:.6f}%")

    # Error distribution histogram
    errors = torch.abs(u_pred - u_exact).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(HERE/'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Predicted vs. exact values scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(u_exact.cpu().numpy(), u_pred.cpu().numpy(), alpha=0.5)
    plt.plot([u_exact.min().cpu(), u_exact.max().cpu()], [u_exact.min().cpu(), u_exact.max().cpu()], 'r--', lw=2)
    plt.xlabel('Exact Solution')
    plt.ylabel('PINN Prediction')
    plt.title('PINN Prediction vs Exact Solution')
    plt.grid(True, alpha=0.3)
    plt.savefig(HERE/'prediction_vs_exact.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'relative_l2': relative_l2.item(),
        'boundary_error': boundary_error.item(),
        'iter_error': iter_error.item()
    }

def visualize_results_2d(model, pinnfunction, R=1.0, device=None, dtype=torch.float32):
    """
    Visualize the solution in the 2-D case (prediction, exact solution and error).
    """
    if model.layers[0].in_features != 2:
        print("Visualization only available for 2D case")
        return

    # Grid points inside the disk of radius R
    x = np.linspace(-R, R, 100)
    y = np.linspace(-R, R, 100)
    X, Y = np.meshgrid(x, y)

    # Keep only the points inside the disk
    mask = X**2 + Y**2 <= R**2
    X_in = X[mask]
    Y_in = Y[mask]

    # Convert to a torch tensor on the target device
    xy = torch.tensor(np.column_stack([X_in, Y_in]), dtype=dtype, device=device)

    # Predictions
    with torch.no_grad():
        u_pred = pinnfunction(model, xy).cpu().numpy().flatten()

    # Exact values
    u_exact = (1 - (X_in**2 + Y_in**2)) / (1 + (X_in**2 + Y_in**2))

    # Plot prediction, exact solution and absolute error side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sc1 = ax[0].scatter(X_in, Y_in, c=u_pred, cmap='viridis')
    ax[0].set_title('PINN Solution')
    ax[0].set_aspect('equal')  # keep the aspect ratio equal
    plt.colorbar(sc1, ax=ax[0])

    sc2 = ax[1].scatter(X_in, Y_in, c=u_exact, cmap='viridis')
    ax[1].set_title('Exact Solution')
    ax[1].set_aspect('equal')
    plt.colorbar(sc2, ax=ax[1])

    error = np.abs(u_pred - u_exact)
    sc3 = ax[2].scatter(X_in, Y_in, c=error, cmap='viridis')
    ax[2].set_title('Absolute Error')
    ax[2].set_aspect('equal')
    plt.colorbar(sc3, ax=ax[2])

    plt.tight_layout()
    plt.savefig(HERE/'pinn_results_improved.png')

def loss_curve(loss_history, error_history, name='loss&error_curve.png'):
    """
    Plot the training loss and exact error curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='loss', color='blue')
    plt.plot(error_history, label='error', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Error')
    plt.title('Training Loss & Exact Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # logarithmic scale for a clearer view of the decrease
    plt.savefig(HERE/name)
    plt.close()
