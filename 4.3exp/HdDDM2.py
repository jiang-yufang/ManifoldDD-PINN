"""
Numerical experiment of Section 4.3 of the paper: the Schwarz
Alternating Method (Algorithm 3.3) with PINNs on the 6-dimensional product manifold
B^3 x S^3 (which has the boundary S^2 x S^3).

The manifold is covered by two overlapping subdomains (charts with a mixed
Euclidean/spherical structure). Two PINNs are trained alternately over 100
outer steps; at each outer step one subdomain is trained for 5000 inner
epochs while the other subdomain's latest solution supplies the interface
(iteration) boundary condition. The whole experiment is repeated 5 times
with different random initializations.

Outputs (created in this folder):
  error_curve/        relative L2 error curves per trial and the mean over trials
  un-u_inf_curve/     ||u^n - u^inf|| curves
  result/             per-trial CSV logs and the mean/std statistics CSV
  models/             trained model checkpoints
  history_preds/      saved prediction histories
  history_error/      saved per-step error histories
"""

import matplotlib.pyplot as plt
from pinn_models2 import Train_pinn2
import torch
from torch import nn
from basic_functions import *
from datetime import datetime
import os
import csv
import pickle
from pathlib import Path
from data_processing import mean_and_std

HERE = Path(__file__).resolve().parent

def get_run_count():
    """Return the number of times the program has been run (from run_count.txt)."""
    count_file = HERE / "run_count.txt"

    # Create the counter file and initialize it to 0 if it does not exist yet.
    if not os.path.exists(count_file):
        with open(count_file, "w") as f:
            f.write("0")
        return 0

    # Read the current count.
    try:
        with open(count_file, "r") as f:
            count = int(f.read().strip())
        return count
    except (ValueError, IOError):
        return 0

def update_run_count():
    """Increment the run counter by one and return the new value."""
    count_file = HERE / "run_count.txt"
    current_count = get_run_count()
    new_count = current_count + 1

    with open(count_file, "w") as f:
        f.write(str(new_count))

    return new_count

def record_test_results(number, n, b, R, hidden_dim, layers, epochs, points, md, device, dtype, act_function,
                        interior_points_rate, interior_weight, opt, lr, lrs, gamma, steps, lr_steps,
                        training_time, d1_er, d2_er, name):
    """
    Append one test result as a row of a CSV file in the result/ folder.
    """
    name = name + '.csv'
    csv_file = HERE / 'result' / name
    if lrs == None:
        lrs = 'None'
    if type(lrs) is float:
        lrs = f"{lrs:.4f}"
    data = {
        'No.': number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n': n,
        'b': b,
        'R': R,
        'hidden_dim': hidden_dim,
        'layers': layers,
        'epochs': epochs,
        'points': points,
        'dtype': str(dtype),
        'model': md,
        'device': str(device),
        'act_function': act_function.__name__,
        'optimizer': opt,
        'interior_points_rate': interior_points_rate,
        'interior_weight': interior_weight,
        'learning_rate': lr,
        'scheduler': lrs,
        'gamma': gamma,
        'steps': steps,
        'lr_steps': lr_steps,
        'time': format(training_time, ".2f"),
        'd1_error': format(d1_er*100, ".6f") + "%",
        'd2_error': format(d2_er*100, ".6f") + "%"
    }

    # Write the header only when the file is created for the first time.
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"Test results recorded to {csv_file}")


def f_function1(x, n, b):
    """
    Source term f = (b+pi^2)sin(pi*y_p)+(b+q)(1-||x||^2)/(1+||x||^2) on D1
    """
    n_=round(n/2)
    norm_x_squared2 = torch.sum(x[:,n_:n]**2, dim=1, keepdim=True)
    return (b + torch.pi**2) * torch.sin(torch.pi * x[:,n_-1:n_]) + (b + n_) * (1 - norm_x_squared2) / (1 + norm_x_squared2)

def f_function2(x, n, b):
    """
    Source term f = (b+pi^2)sin(pi*y_p)+(b+q)(-1+||x||^2)/(1+||x||^2) on D2
    """
    n_=round(n/2)
    norm_x_squared2 = torch.sum(x[:,n_:n]**2, dim=1, keepdim=True)
    return (b + torch.pi**2) * torch.sin(torch.pi * x[:,n_-1:n_]) + (b + n_) * (-1 + norm_x_squared2) / (1 + norm_x_squared2)

def boundary_condition_training(model_b):
    """
    Build the iteration boundary condition for one subdomain.

    The interface data is provided by the *other* subdomain's latest model:
    its prediction at the interface is transported back through the chart
    (only the spherical coordinates are scaled by 1/R^2) and used as the
    boundary values of the current subdomain.
    """
    if model_b is None:
        def boundary_function(x,R):
            return 0
    else:
        def boundary_function(x,R):
            x_ = x.detach().clone()
            with torch.no_grad():
                n = x_.shape[1]
                n_ = round(n/2)
                x_[:,n_:n] = x_[:,n_:n] / R**2
                boundary = model_b(x_)
            return boundary
    return boundary_function

def exact_solution1(x):
    """
    Exact solution u = sin(pi*y_p)+(1-||x||^2)/(1+||x||^2) on D1
    """
    n = x.shape[1]
    n_ = round(n/2)
    norm_x_squared2 = torch.sum(x[:,n_:n]**2, dim=1, keepdim=True)
    return torch.sin(torch.pi * x[:,n_-1:n_]) + (1-norm_x_squared2) / (1+norm_x_squared2)

def exact_solution2(x):
    """
    Exact solution u = sin(pi*y_p)+(-1+||x||^2)/(1+||x||^2) on D2
    """
    n = x.shape[1]
    n_ = round(n/2)
    norm_x_squared2 = torch.sum(x[:,n_:n]**2, dim=1, keepdim=True)
    return torch.sin(torch.pi * x[:,n_-1:n_]) + (-1+norm_x_squared2) / (1+norm_x_squared2)

d1_ers = []
d2_ers = []
test_d1s = []
test_d2s = []
numbers = []
# Repeat the whole experiment 5 times with different random initializations.
for k in range(5):
    run_count = update_run_count()
    numbers.append(run_count)
    print(f'Record No.{run_count}')
    # First and last run numbers of this batch of 5 trials (used for the mean plots).
    if k==0:
        st = run_count
    elif k == 4:
        ed = run_count
    save_dir = str(HERE / 'models')
    save_his = str(HERE / 'history_preds')
    save_his_er = str(HERE / 'history_error')
    n = 6
    b = 0.0
    R =1.2
    hidden_dim = 500
    layers = 2
    epochs = 5000
    points = 2000
    md = 'Res'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    dtype = torch.float32
    act_function = nn.Tanh
    interior_points_rate = 0.1
    interior_weight = 0.005
    opt = 'Adam'
    lr = 0.001
    lrs = 'exp'
    gamma = 0.9999
    steps = 100
    lr_steps = 0.9

    d1_er = []
    d2_er = []
    total_time = 0

    # Build the two subdomain networks (subdomains D1 and D2).
    if md == 'PINN':
        model1 = PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)
        model2 = PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)
    elif md == 'Res':
        model1 = Res_PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)
        model2 = Res_PINN(input_dim=n, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device)

    # Each subdomain uses the other subdomain's latest model as the
    # iteration boundary condition (initialized here with the random models).
    task1 = Train_pinn2(n, b, R, hidden_dim, layers, epochs, 
        points, model1, device, dtype, act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma,
        f_function=f_function1, boundary_condition=boundary_condition_training(model2), exact_solution=exact_solution1)
    task2 = Train_pinn2(n, b, R, hidden_dim, layers, epochs, 
        points, model2, device, dtype, act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma,
        f_function=f_function2, boundary_condition=boundary_condition_training(model1), exact_solution=exact_solution2)
    # Relative L2 error of the random initialization (outer step 0).
    task1.test()
    d1_er.append(task1.detailed_results['relative_l2'])
    task2.test()
    d2_er.append(task2.detailed_results['relative_l2'])

    # Fixed test points, independent of the points used during training. The
    # solution values at these points are recorded at every outer step (one
    # row per step in test_d1/test_d2) instead of storing the whole model per
    # step; they are used later to compute the ||u^n - u^inf|| curves.
    test_points_d1 = task1.get_interior_points(10000)
    test_points_d2 = task2.get_interior_points(10000)
    test_d1 = torch.zeros(steps+1,10000, device=device, dtype=dtype)
    test_d2 = torch.zeros(steps+1,10000, device=device, dtype=dtype)
    with torch.no_grad():
        test_d1[0] = model1(test_points_d1).flatten()
        test_d2[0] = model2(test_points_d2).flatten()
    lr_ = lr
    # Outer iteration: alternately train the two subdomains; each one uses the
    # other's latest solution as the interface boundary condition.
    for step in range(steps):
        task1.train()
        task1.test()
        total_time += task1.training_time
        d1_er.append(task1.detailed_results['relative_l2'])
        task2 = Train_pinn2(n, b, R, hidden_dim, layers, epochs, 
            points, task2.model, device, dtype, act_function, interior_points_rate, interior_weight, opt, lr_, lrs, gamma,
            f_function=f_function2, boundary_condition=boundary_condition_training(task1.model), exact_solution=exact_solution2)
        task2.train()
        task2.test()
        total_time += task2.training_time
        d2_er.append(task2.detailed_results['relative_l2'])
        task1.boundary_condition = boundary_condition_training(task2.model)
        with torch.no_grad():
            # Record the solution values at the fixed test points (one row per outer step).
            test_d1[step+1]=task1.model(test_points_d1).flatten()
            test_d2[step+1]=task2.model(test_points_d2).flatten()
        print(f"Step {step}, D1 exact error: {d1_er[-1]:.6f}, D2 exact error: {d2_er[-1]:.6f}, time: {task1.training_time+task2.training_time:.2f}")

        # Decay the learning rate at every outer step.
        lr_ = lr_ * lr_steps
        task1 = Train_pinn2(n, b, R, hidden_dim, layers, epochs, 
            points, task1.model, device, dtype, act_function, interior_points_rate, interior_weight, opt, lr_, lrs, gamma,
            f_function=f_function1, boundary_condition=boundary_condition_training(task2.model), exact_solution=exact_solution1)

    # Save the trained models of both subdomains.
    model_path = os.path.join(save_dir, f'models_{run_count}.pth')
    models = {
        'model1':task1.model.state_dict(),
        'model2':task2.model.state_dict()
    }
    torch.save(models, model_path)

    # Save the prediction history on the test points (one row per outer step).
    history_path = os.path.join(save_his,f'historys_{run_count}.pth')
    historys = {
        'test_d1': test_d1,
        'test_d2': test_d2,
    }
    torch.save(historys, history_path)

    # Package the per-step error histories of both subdomains into one dict.
    with open(HERE/f'history_error/d_er_{run_count}.pkl', 'wb') as f:
        data_to_save = {'d1_er': d1_er, 'd2_er': d2_er}
        pickle.dump(data_to_save, f)

    # Relative L2 error curves of this trial.
    plt.figure(figsize=(10, 6))
    plt.plot(d1_er, label='d1', color='blue')
    plt.plot(d2_er, label='d2', color='red')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # logarithmic scale for a clearer view of the decrease
    plt.savefig(HERE/f'error_curve/loss&error_curve_improved_{run_count}.png')
    plt.close()

    der_max = [max(value) for value in zip(*[d1_er,d2_er])]

    plt.figure(figsize=(10, 6))
    plt.plot(der_max, color='blue')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'error_curve/loss&error_curve_improved_max_{run_count}.png')
    plt.close()

    # ||u^n - u^inf|| curves: relative L2 error of u^n w.r.t. the converged
    # solution u^inf := u^100 (step 100).
    test_d1 = torch.sqrt(torch.sum((test_d1 - test_d1[steps])**2,dim=1))/torch.sqrt(torch.sum((test_d1[steps])**2))
    test_d2 = torch.sqrt(torch.sum((test_d2 - test_d2[steps])**2,dim=1))/torch.sqrt(torch.sum((test_d2[steps])**2))
    test_d1 = test_d1.tolist()
    test_d2 = test_d2.tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(test_d1[0:round(steps*0.5)], label='d1', color='blue')
    plt.plot(test_d2[0:round(steps*0.5)], label='d2', color='red')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_all_{run_count}.png')
    plt.close()

    test_max = [max(value) for value in zip(*[test_d1,test_d2])]

    plt.figure(figsize=(10, 6))
    plt.plot(test_max[0:round(steps*0.5)], color='blue')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_max_{run_count}.png')
    plt.close()

    d1_ers.append(d1_er)
    d2_ers.append(d2_er)
    test_d1s.append(test_d1)
    test_d2s.append(test_d2)

    record_test_results(run_count, n, b, R, hidden_dim, layers, epochs, points, md, device, dtype, act_function,
                            interior_points_rate, interior_weight, opt, lr, lrs, gamma, steps, lr_steps,
                            total_time, d1_er[steps], d2_er[steps], 'hdddm2_test_results')

# Mean curves over the 5 trials (element-wise average).
d1_ers = [sum(value)/len(value) for value in zip(*d1_ers)]
d2_ers = [sum(value)/len(value) for value in zip(*d2_ers)]
test_d1s = [sum(value)/len(value) for value in zip(*test_d1s)]
test_d2s = [sum(value)/len(value) for value in zip(*test_d2s)]
plt.figure(figsize=(10, 6))
plt.plot(d1_ers, label='d1', color='blue')
plt.plot(d2_ers, label='d2', color='red')
plt.xlabel('step')
plt.ylabel('relative_l2')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'error_curve/loss&error_curve_improved_mean_all_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
er_maxs = [max(value) for value in zip(*[d1_ers,d2_ers])]
plt.plot(er_maxs, color='blue')
plt.xlabel('step')
plt.ylabel('relative_l2')
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'error_curve/loss&error_curve_improved_mean_max_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(test_d1s[0:round(steps*0.5)], label='d1', color='blue')
plt.plot(test_d2s[0:round(steps*0.5)], label='d2', color='red')
plt.xlabel('step')
plt.ylabel('un-u_inf relative_l2')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_mean_all_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
test_maxs = [max(value) for value in zip(*[test_d1s,test_d2s])]
plt.plot(test_maxs[0:round(steps*0.5)], color='blue')
plt.xlabel('step')
plt.ylabel('un-u_inf relative_l2')
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_mean_max_{st}-{ed}.png')
plt.close()

# Compute the mean and standard deviation at steps 0, 5, 10, 15, 20 and 100.
mean_and_std(list(range(st,ed+1)))
