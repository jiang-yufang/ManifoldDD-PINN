"""
Smoke test for the DDM-PINN experiments.

Runs every main program (HdDDM1.py / HdDDM2.py / HdDDM3.py / HdDDM4.py)
end-to-end with drastically reduced parameters (30 inner epochs, 5 outer
steps, 300 sampled points, 64-wide networks) to verify that the pipeline
runs without errors and produces its expected output files. The number of
trials is intentionally kept at the default 5: the main programs only assign
their `st`/`ed` run numbers on the first/last trial, and keeping 5 trials
also exercises the mean/std statistics path.

NOTE: this only checks the plumbing. Reproducing the paper results requires
the full configuration and a CUDA GPU (see README). The official code is
never modified: each experiment is copied to a temporary work directory and
only the copies are shrunk.

Usage:
    python smoke_test.py             # run all five experiments
    python smoke_test.py 4.1exp      # run only one experiment folder
    python smoke_test.py --keep      # keep the temporary work directory
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ('4.1exp', 'HdDDM1.py'),
    ('4.2exp', 'HdDDM1.py'),
    ('4.3exp', 'HdDDM2.py'),
    ('4.4exp_serial', 'HdDDM3.py'),
    ('4.4exp_parallel', 'HdDDM4.py'),
]

# Shrinking patches applied to a *copy* of each main program.
# NOTE: the trial count (`for k in range(5)`) is intentionally NOT patched:
# `st`/`ed` are only assigned on the first/last trial, and keeping 5 trials
# exercises the mean/std statistics path as well.
PATCHES = [
    ('    epochs = 5000', '    epochs = 30'),       # inner epochs
    ('    points = 2000', '    points = 300'),      # sampled points
    ('    points = 5000', '    points = 300'),      # sampled points (4.2)
    ('    hidden_dim = 500', '    hidden_dim = 64'),
    ('    steps = 100', '    steps = 5'),           # outer steps
]

REQUIRED = ['    epochs = 30']

EXPECTED_OUTPUTS = [
    'result/output.csv',
    'error_curve',
    'un-u_inf_curve',
    'models',
    'history_preds',
    'history_error',
]


def patch_main(src_path, dst_path):
    """Shrink the parameters of a main program copy (never touches the source)."""
    with open(src_path, encoding='utf-8') as f:
        text = f.read()
    for pat, rep in PATCHES:
        if pat in text:
            text = text.replace(pat, rep)
    for req in REQUIRED:
        if req not in text:
            raise SystemExit(f'FAILED to patch {src_path}: required pattern {req!r} missing')
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(text)


def check_outputs(exp_dir):
    """Return the list of missing expected outputs (empty means all present)."""
    missing = []
    for name in EXPECTED_OUTPUTS:
        path = os.path.join(exp_dir, name)
        if os.path.isdir(path):
            if not any(f for f in os.listdir(path) if f != '.gitkeep'):
                missing.append(f'{name}/ (empty)')
        elif not os.path.isfile(path):
            missing.append(name)
    # a per-trial CSV log should also exist
    result_dir = os.path.join(exp_dir, 'result')
    if os.path.isdir(result_dir):
        csvs = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
        if len(csvs) < 2:
            missing.append('result/*.csv (per-trial log + statistics)')
    else:
        missing.append('result/')
    return missing


def run_one(workdir, exp, main, keep, verbose):
    label = f'{exp}/{main}'
    print(f'=== {label} ===')
    src_dir = os.path.join(ROOT, exp)
    dst_dir = os.path.join(workdir, exp)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('__pycache__'))
    patch_main(os.path.join(src_dir, main), os.path.join(dst_dir, main))
    print(f'  work dir: {dst_dir}')

    try:
        proc = subprocess.run(
            [sys.executable, main], cwd=dst_dir,
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print(f'  [FAIL] {label}: timed out after 1 hour')
        return False

    if proc.returncode != 0:
        tail = proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:]
        print(f'  [FAIL] {label}: exited with code {proc.returncode}')
        print('  --- stderr/stdout tail ---')
        print(tail)
        print('  --------------------------')
        return False

    missing = check_outputs(dst_dir)
    if missing:
        print(f'  [FAIL] {label}: ran OK but missing outputs: {missing}')
        return False

    print(f'  [PASS] {label}: ran end-to-end and all outputs were produced')
    if verbose:
        for f in sorted(os.listdir(os.path.join(dst_dir, 'result'))):
            print(f'    result/{f}')
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    keep = '--keep' in sys.argv
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    only = args[0] if args else None

    # Pre-check: torch must be importable with the same interpreter.
    check = subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'],
                           capture_output=True, text=True)
    if check.returncode != 0:
        print('torch is not installed for this Python interpreter.')
        print('Install the requirements first, e.g.:  pip install -r requirements.txt')
        print('(for CUDA GPU support see the README installation section)')
        return 2
    print(f'torch OK (CUDA available: {check.stdout.strip()})')

    workdir = tempfile.mkdtemp(prefix='hdddm_smoke_')
    print(f'work directory: {workdir}')
    results = []
    try:
        for exp, main in EXPERIMENTS:
            if only and exp != only:
                continue
            if not os.path.isdir(os.path.join(ROOT, exp)):
                print(f'[SKIP] {exp}: folder not found')
                continue
            results.append((f'{exp}/{main}', run_one(workdir, exp, main, keep, verbose)))
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
            print('temporary work directory removed (use --keep to retain it)')

    print()
    print('===== SUMMARY =====')
    ok = 0
    for label, passed in results:
        print(f'  [{"PASS" if passed else "FAIL"}] {label}')
        ok += int(passed)
    print(f'{ok}/{len(results)} experiments passed')
    return 0 if ok == len(results) and results else 1


if __name__ == '__main__':
    sys.exit(main())
