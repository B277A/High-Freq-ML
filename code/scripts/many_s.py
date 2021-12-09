import os
import numpy as np


def job_func(job_name):
    s = '#!/bin/bash\n'
    s += '#SBATCH --job-name={}\n'.format(job_name)
    s += '#SBATCH --partition=q36\n'
    s += '#SBATCH --mem=20G\n'
    s += '#SBATCH --nodes=1\n'
    s += '#SBATCH --time=120:00:00\n'
    s += '#SBATCH --ntasks-per-node=20\n'
    s += 'cd $SLURM_SUBMIT_DIR\n'
    s += 'source ~/.bashrc\n'
    s += 'python test_server.py \n'
    return s

cwd = str(os.getcwd())

ram_allocated = 20
job_name = 'testing_1'
f = open('slurm.job', 'w')
f.write(job_func(job_name))
f.close()
os.system('sbatch --mem={}G slurm.job'.format(ram_allocated))
os.chdir(cwd)


