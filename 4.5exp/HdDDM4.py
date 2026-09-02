"""
Numerical experiment of Section 4.5 of the paper: the parallel domain
decomposition method (Algorithm 3.2) with PINNs on the complex projective
space CP^3 (complex dimension 3, real dimension 6).

CP^3 is decomposed into 4 overlapping subdomains. Unlike the sequential DDM,
all subdomains are trained simultaneously at every outer step; the solutions
of the neighboring subdomains are combined with a partition of unity to form
the iteration boundary condition. The whole experiment is repeated 5 times
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
from pinn_models4 import Train_pinn3
import torch
from torch import nn
from basic_functions import *
from datetime import datetime
import os
import csv
from copy import deepcopy
import pickle
from pathlib import Path
from data_processing import mean_and_std

HERE = Path(__file__).resolve().parent

def get_run_count():
    """Return the number of times the program has been run (from run_count.txt)."""
    count_file = "run_count.txt"

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
    count_file = "run_count.txt"
    current_count = get_run_count()
    new_count = current_count + 1

    with open(count_file, "w") as f:
        f.write(str(new_count))

    return new_count

def record_test_results(number, n, b, a, R, s, hidden_dim, layers, epochs, points, md, device, dtype, act_function,
                        interior_points_rate, interior_weight, opt, lr, lrs, gamma, steps, lr_steps,
                        training_time, di_er, name):
    """
    Append one test result as a row of a CSV file in the result/ folder.
    """
    csv_file = 'result/' + name + '.csv'
    if lrs == None:
        lrs = 'None'
    if type(lrs) is float:
        lrs = f"{lrs:.4f}"
    data = {
        'No.': number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n': n,
        'b': b,
        'a': str(a.tolist()),
        'R': R,
        's': format(s, '.2f'),
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
        'dmax_error': format(max(di_er)*100, ".6f") + "%",
    }

    # Write the header only when the file is created for the first time.
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"Test results recorded to {csv_file}")

def get_w_norm2(x):
    """Squared moduli |w_j|^2 of each complex coordinate of the chart point x."""
    n = x.shape[1]  # spatial dimension
    N = x.shape[0]
    n_ = round(n/2)
    w_norm2 = torch.zeros(N,n_).to(device=x.device,dtype=x.dtype)
    for i in range(n_):
        w_norm2[:,i:i+1] = x[:,i:i+1]**2 + x[:,n_+i:n_+i+1]**2
    return w_norm2

def domain_transport(x,i,j):
    '''
    Map a point of subdomain D_i (chart i) into subdomain D_j (chart j)
    through the global homogeneous coordinates of CP^k.
    '''
    m,n = x.shape
    n_ = round(n/2)
    x_cp = torch.complex(x[:,:n_], x[:,n_:])
    x_global = torch.zeros(m,n_+1).to(device=x_cp.device, dtype=x_cp.dtype)
    i_mask = torch.full((n_+1,), True, dtype=torch.bool, device=x_cp.device)
    i_mask[i] = False
    j_mask = torch.full((n_+1,), True, dtype=torch.bool, device=x_cp.device)
    j_mask[j] = False
    x_global[:,i_mask] = x_cp
    x_global[:,i:i+1] = 1
    x_global = x_global / x_global[:,j:j+1]
    x_ = torch.zeros(m,n).to(device=x.device, dtype=x.dtype)
    x_[:,:n_] = x_global[:,j_mask].real
    x_[:,n_:] = x_global[:,j_mask].imag
    return x_

def f_function_i(i):
    def f_function(x, n, b, a):
        """
        Source term f = (4n + 4 + b)*u - 4Σa_i on Di
        """
        norm_x_squared = torch.sum(x**2, dim=1, keepdim=True)
        w_norm2 = get_w_norm2(x)
        i_mask = torch.full((n+1,), True, dtype=torch.bool, device=a.device)
        i_mask[i] = False
        return (4*n + b + 4) * (a[i] + torch.sum(a[i_mask] * w_norm2, dim=1, keepdim=True)) / (1 + norm_x_squared) - 4 * torch.sum(a)
    return f_function

def boundary_condition_training_i(i, s, model_list):
    """
    Build the iteration boundary condition for subdomain D_i (parallel DDM).

    The overlap size is controlled by s (R' = (R-1)*s + 1). For every point
    of D_i that lies in the overlap with a neighbor, the boundary value is a
    partition-of-unity combination of the neighbors' solutions: the weights
    {sigma_p} are products of the scaled distance functions, normalized so
    that they sum to one, and the neighbor values are transported to chart i.
    """
    if model_list is None:
        def boundary_function(x,R,a):
            return 0
    else:
        def boundary_function(x,R,a):
            R = (R - 1) * s + 1
            x_ = x.detach().clone()
            n = round(x_.shape[1]/2)
            m = x_.shape[0]
            sigma = torch.zeros(m,n).to(device = x.device, dtype=x.dtype)
            us = torch.zeros(m,n).to(device = x.device, dtype=x.dtype)
            t = 0
            with torch.no_grad():
                w_norm2 = get_w_norm2(x_)
                for p in (j for j in range(n+1) if j != i):
                    p_mask = torch.full((n,), True, dtype=torch.bool)
                    p_mask[t] = False
                    using_mask = torch.all(torch.cat((torch.ones(x_.shape[0] , 1, device=x_.device, dtype=x_.dtype),
                                                      w_norm2[:,p_mask]), dim=1) < R**2 * w_norm2[:,t:t+1], dim=1)
                    x_in_p = x_[using_mask,:]
                    xp = domain_transport(x_in_p,i,p)
                    model_b = model_list[p]
                    us[using_mask,t:t+1] = model_b(xp)
                    sigma[using_mask,t:t+1] = torch.prod((1 - get_w_norm2(xp) / R**2).clamp(min=0),
                                            dim=1, keepdim=True)**3
                    t = t + 1
                sigma = sigma / torch.sum(sigma, dim=1, keepdim=True)
                boundary = torch.sum(sigma * us, dim=1, keepdim=True)
            return boundary
    return boundary_function

def exact_solution_i(i):
    def exact_solution(x, a):
        """
        Exact solution u = (ai + Σaj*|wj|^2) / (1 + ||w||^2) on Di
        """
        w_norm2 = get_w_norm2(x)
        i_mask = torch.full((a.shape[0],), True, dtype=torch.bool, device=a.device)
        i_mask[i] = False
        return (a[i] + torch.sum(a[i_mask] * w_norm2, dim=1, keepdim=True)) / (1 + torch.sum(w_norm2, dim=1, keepdim=True))
    return exact_solution

n = 3

di_ers = [[] for _ in range(n+1)]
test_dis = [[] for _ in range(n+1)]
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
    save_dir = 'models'
    save_his = 'history_preds'
    save_his_er = 'history_error'
    b = 4.0
    a = torch.tensor([1.,2.,-1.,-2.])
    R = 1.2
    s = 0.1
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
    a = a.to(device=device, dtype=dtype)
    act_function = nn.Tanh
    interior_points_rate = 0.5
    interior_weight = 0.003
    opt = 'Adam'
    lr = 0.001
    lrs = 'exp'
    gamma = 0.9999
    steps = 100
    lr_steps = 0.9

    di_er = [[] for _ in range(n+1)]
    total_time = 0
    model_list = []
    task_i = []
    # Build one network per subdomain (4 subdomains for CP^3).
    for i in range(n+1):
        if md == 'PINN':
            model_list.append(PINN(input_dim=n*2, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device))
        elif md == 'Res':
            model_list.append(Res_PINN(input_dim=n*2, hidden_dim=hidden_dim, layers=layers, dtype=dtype, act_function = act_function).to(device))

    # Each subdomain uses the neighboring subdomains' latest models as the
    # iteration boundary condition (initialized here with the random models).
    for i in range(n+1):
        task_i.append(Train_pinn3(n, b, a, R, hidden_dim, layers, epochs, 
            points, model_list[i], device, dtype, act_function, interior_points_rate, interior_weight, opt, lr, lrs, gamma,
            f_function=f_function_i(i), boundary_condition=boundary_condition_training_i(i, s, model_list), exact_solution=exact_solution_i(i)))
    # Relative L2 error of the random initialization (outer step 0).
    for i in range(n+1):
        task_i[i].test()
        di_er[i].append(task_i[i].detailed_results['relative_l2'])

    # Fixed test points, independent of the points used during training. The
    # solution values at these points are recorded at every outer step (one
    # row per step in test_di) instead of storing the whole model per step;
    # they are used later to compute the ||u^n - u^inf|| curves.
    test_points_di = [[] for _ in range(n+1)]
    test_di = [[] for _ in range(n+1)]
    for i in range(n+1):
        test_points_di[i] = task_i[i].get_interior_points(10000)
        test_di[i] = torch.zeros(steps+1,10000, device=device, dtype=dtype)
        with torch.no_grad():
            test_di[i][0] = model_list[i](test_points_di[i]).flatten()
    lr_ = lr
    # Freeze a copy of every subdomain model. model_list provides the interface
    # data (via the partition of unity) during the parallel iteration, while
    # task_i[i].model is the one being trained.
    for i in range(n+1):
        model_list[i] = deepcopy(task_i[i].model)
        for param in model_list[i].parameters():
            param.requires_grad = False
    # Outer iteration (parallel DDM): update all subdomains simultaneously,
    # then refresh the frozen neighbor models.
    for step in range(steps):
        # Refresh every subdomain's iteration boundary condition from the
        # frozen neighbor models (parallel update, no dependency between
        # the subdomains within one step).
        for i in range(n+1):
            task_i[i].boundary_condition = boundary_condition_training_i(i, s, model_list)
        for i in range(n+1):
            task_i[i].lr = lr_
            task_i[i].train()
            task_i[i].test()
            total_time += task_i[i].training_time
            di_er[i].append(task_i[i].detailed_results['relative_l2'])
            with torch.no_grad():
                # Record the solution values at the fixed test points (one row per outer step).
                test_di[i][step+1]=task_i[i].model(test_points_di[i]).flatten()
            print(f"Step {step}, D{str(i)} exact error: {di_er[i][step]:.6f}, time: {task_i[i].training_time:.2f}")

        # Decay the learning rate at every outer step.
        lr_ = lr_ * lr_steps
        # Freeze the updated models for the next outer step.
        for i in range(n+1):
            model_list[i] = deepcopy(task_i[i].model)

    # Save the trained models of all subdomains.
    model_path = os.path.join(save_dir, f'models_{run_count}.pth')
    modeli = [[] for _ in range(n+1)]
    for i in range(n+1):
        modeli[i] = task_i[i].model.state_dict()
    models = {
        'modeli':modeli
    }
    torch.save(models, model_path)

    # Save the prediction history on the test points (one row per outer step).
    history_path = os.path.join(save_his,f'historys_{run_count}.pth')
    historys = {
        'test_di': test_di
    }
    torch.save(historys, history_path)
    # Package the per-step error histories of all subdomains into one dict.
    with open(HERE/f'history_error/d_er_{run_count}.pkl', 'wb') as f:
        data_to_save = {'di_er': di_er}
        pickle.dump(data_to_save, f)
    colors = plt.cm.tab10(range(n+1))  # up to 10 colors
    # Relative L2 error curves of this trial.
    plt.figure(figsize=(10, 6))
    for i in range(n+1):
        plt.plot(di_er[i], label='d'+str(i), color=colors[i])

    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # logarithmic scale for a clearer view of the decrease
    plt.savefig(HERE/f'error_curve/loss&error_curve_improved_all_{run_count}.png')
    plt.close()

    der_max = [max(col) for col in zip(*di_er)]

    plt.figure(figsize=(10, 6))
    plt.plot(der_max, label='er_max')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'error_curve/loss&error_curve_improved_max_{run_count}.png')
    plt.close()

    # ||u^n - u^inf|| curves: relative L2 error of u^n w.r.t. the converged
    # solution u^inf := u^100 (step 100).
    for i in range(n+1):
        test_di[i] = torch.sqrt(torch.sum((test_di[i] - test_di[i][steps-1])**2,dim=1))/torch.sqrt(torch.sum((test_di[i][steps-1])**2))
        test_di[i] = test_di[i].tolist()

    plt.figure(figsize=(10, 6))
    for i in range(n+1):
        plt.plot(test_di[i][0:round(steps*0.5)], label='d'+str(i), color=colors[i])
    plt.xlabel('step')
    plt.ylabel('un-u_inf relative_l2')
    plt.title('un-u_inf relative_l2 Over Step')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_all_{run_count}.png')
    plt.close()

    test_max = [max(value) for value in zip(*test_di)]

    plt.figure(figsize=(10, 6))
    plt.plot(test_max[0:round(steps*0.5)], color='blue')
    plt.xlabel('step')
    plt.ylabel('relative_l2')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_max_{run_count}.png')
    plt.close()

    for i in range(n+1):
        di_ers[i].append(di_er[i])
        test_dis[i].append(test_di[i])

    record_test_results(run_count, n, b, a, R, s, hidden_dim, layers, epochs, points, md, device, dtype, act_function,
                            interior_points_rate, interior_weight, opt, lr, lrs, gamma, steps, lr_steps,
                            total_time, [er[step] for er in di_er], 'hdddm4_test_results')

# Mean curves over the 5 trials (element-wise average).
for i in range(n+1):
    di_ers[i] = [sum(value)/len(value) for value in zip(*di_ers[i])]
    test_dis[i] = [sum(value)/len(value) for value in zip(*test_dis[i])]

plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(range(n+1))  # up to 10 colors
for i in range(n+1):
    plt.plot(di_ers[i], label='d'+str(i), color=colors[i])

plt.xlabel('step')
plt.ylabel('relative_l2')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'error_curve/loss&error_curve_improved_mean_all_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
er_maxs = [max(value) for value in zip(*di_ers)]
plt.plot(er_maxs, color='blue')
plt.xlabel('step')
plt.ylabel('relative_l2')
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'error_curve/loss&error_curve_improved_mean_max_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
for i in range(n+1):
    plt.plot(test_dis[i][0:round(steps*0.5)], label='d'+str(i), color=colors[i])

plt.xlabel('step')
plt.ylabel('un-u_inf relative_l2')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_mean_all_{st}-{ed}.png')
plt.close()

plt.figure(figsize=(10, 6))
test_maxs = [max(value) for value in zip(*test_dis)]
plt.plot(test_maxs[0:round(steps*0.5)], color='blue')
plt.xlabel('step')
plt.ylabel('un-u_inf relative_l2')
plt.grid(True)
plt.yscale('log')
plt.savefig(HERE/f'un-u_inf_curve/un-u_inf__relative_l2_Over_Step_mean_max_{st}-{ed}.png')
plt.close()

mean_and_std(list(range(st,ed+1)))
