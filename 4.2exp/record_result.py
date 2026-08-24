import csv
import os
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def record_pinn_test_results(n, b, R, hidden_dim, layers, epochs, points, md, device, dtype, act_function,
                        interior_points_rate, interior_weight, opt, lr, lrs, gamma,
                        training_time, relative_error, name):
    """
    Append one test result as a row of a CSV file in the experiment folder.
    """
    csv_file = name + '.csv'
    csv_file = HERE/csv_file
    if lrs == None:
        lrs = 'None'
    data = {
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
        'time': format(training_time, ".2f"),
        'relative_error': format(relative_error*100, ".6f") + "%"
    }

    # Write the header only when the file is created for the first time.
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"Test results recorded to {csv_file}")
