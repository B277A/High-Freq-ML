import os
import numpy as np

ram_allocated = 20

def job_func(job_name, i, j):
    s = '#!/bin/bash\n'
    s += '#SBATCH --job-name={}\n'.format(job_name)
    s += '#SBATCH --partition=q36\n'
    s += '#SBATCH --mem=20G\n'
    s += '#SBATCH --nodes=1\n'
    s += '#SBATCH --time=200:00:00\n'
    s += '#SBATCH --ntasks-per-node=1\n'
    s += 'cd $SLURM_SUBMIT_DIR\n'
    s += 'source ~/.bashrc\n'
    s += 'python test_main.py {}\n'.format(str(i) + ',' + str(j))
    return s

cwd = str(os.getcwd())

model_list = ['LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso', 'Enet']

for i in range(0,len(model_list)):
    for j in range(0,12):
        ram_allocated = 20
        newdir = '{}_{}_script'.format(model_list[i],j)
        os.mkdir(newdir)
        print(newdir)
        os.chdir(newdir)
        f = open('slurm.job', 'w')
        f.write(job_func(newdir, i, j))
        f.close()
        os.system('cp ../test_main.py .')
        os.system('sbatch --mem={}G slurm.job'.format(ram_allocated))
        os.chdir(cwd)


