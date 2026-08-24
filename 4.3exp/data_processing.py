import pickle
import math
from pathlib import Path
import csv
import os

HERE = Path(__file__).resolve().parent

d1_ers = []
d2_ers = []
der_max = []


def mean_and_std(index):
    """
    Aggregate the per-step error histories saved by the main program.

    For each trial index in `index`, loads history_error/d_er_<k>.pkl, takes the
    maximum relative L2 error over the subdomains at every outer step, and then
    computes the mean and standard deviation (over the trials) at outer steps
    0, 5, 10, 15, 20 and 100. The results are appended to result/output.csv.
    """
    for k in index:
        with open(HERE/f'history_error/d_er_{k}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
            d1_ers.append(loaded_data['d1_er'])
            d2_ers.append(loaded_data['d2_er'])

    # Max relative L2 error over the two subdomains, step by step, per trial.
    for k in range(len(d1_ers)):
        der_max.append([max(value) for value in zip(*[d1_ers[k],d2_ers[k]])])

    der_max_mean = [sum(value)/len(value) for value in zip(*der_max)]
    der_max_bias = [[(der_max[i][j] - der_max_mean[j])**2 for j in range(len(der_max[0]))] for i in range(len(der_max))]
    der_max_bias = [math.sqrt(sum(value)/len(value)) for value in zip(*der_max_bias)]
    print(der_max_mean[0:21:5]+[der_max_mean[-1]])
    print(der_max_bias[0:21:5]+[der_max_bias[-1]])

    file_exists = os.path.isfile(HERE/'result'/'output.csv')
    with open(HERE/'result'/'output.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['step']+[0,5,10,15,20,100])
        writer.writerow([f'{index[0]}-{index[-1]} mean']+der_max_mean[0:21:5]+[der_max_mean[-1]])
        writer.writerow([f'{index[0]}-{index[-1]} std dev']+der_max_bias[0:21:5]+[der_max_bias[-1]])

    print("Data written to output.csv")
