import torch
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
def detailed_evaluation(model, pinnfunction, get_interior_points, get_boundary_points, exact_solution):
    """
    Evaluate the accuracy of a PINN model in detail.

    Computes the relative L2 error on a large set of interior points and the
    relative L2 error on the physical boundary, and saves two diagnostic plots.
    """
    # Generate a large set of test points
    x_test = get_interior_points(100000)
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

    # Boundary condition check
    x_boundary = get_boundary_points(100000)
    with torch.no_grad():
        u_boundary_pred = pinnfunction(model, x_boundary)
    u_boundary_exact = exact_solution(x_boundary)

    boundary_error = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_exact)**2))/ torch.sqrt(torch.mean(u_boundary_exact**2))
    print(f"Boundary condition relative L2 error: {boundary_error.item()*100:.6f}%")

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
        'boundary_error': boundary_error.item()
    }

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
