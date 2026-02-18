"""This script submits slurm jobs to run benchmarking simulations used in the
paper. The results are saved in the `artifact/data/` directory.

The script generates a large number of "experiment configurations", each of
which corresponds to one SWIPER-SIM simulation (by running
`artifact/_simulate.py`). The script then submits one or more arrays of SLURM
jobs to run these simulations. Each job reads one experiment configuration from
the `config.json` file and runs the corresponding simulation.

Requires a compute cluster with SLURM and a valid Python installation. Python
environment must be configured properly first. The four variables at the start
of this file must be configured to match the user's system. The rest of the
script does not need to be modified to reproduce the results in the paper, but
may require modifications if the user's system has a shorter job time limit or a
limit on the number of jobs that can be submitted at once.

This script is adapted from `slurm/run_simulation.py` and `slurm/submit_jobs.py`
in the original repository.

This script can take between 1 minute and an hour to submit all jobs,
depending on configuration of `submission_delay` and `max_tasks_per_job`. The
SLURM jobs themselves will finish in a few hours."""

import os, sys
sys.path.append('.')
import shutil
import json
import math
import subprocess
import datetime as dt
import time
from functools import reduce
import numpy as np

################################################################################
# CONFIGURATION
################################################################################

dry_run = False # if True, don't actually submit jobs, just print sbatch commands

# Required configuration
slurm_account = 'martonosi' # argument to `--account` in `sbatch`
slurm_partition = 'n/a?' # argument to `--partition` in `sbatch` # I don't think I need a partition?
path_to_python_env = 'swiper' # argument to `conda activate` in sbatch script
path_to_swiper = '/home/jl1543/swiper' # path to the root of this repository

# Optional configuration
submission_delay = dt.timedelta(minutes=10) # gap between submissions of job arrays
max_tasks_per_job = 800 # this is the maximum number of tasks a job array can take/handle/do

################################################################################
# Check variables are configured/changed
if not dry_run and any([x == 'CHANGE_ME' for x in [slurm_account, slurm_partition, path_to_python_env, path_to_swiper]]):
    raise ValueError('Please configure the variables at the top of this script.')

if __name__ == '__main__':
    cur_time = dt.datetime.now()

    # USER SETTING: maximum job duration
    # set this to slightly longer than 1 hour because of della constraint (if 1 hr or less, will be put on quick queue w/ limited number of jobs, which isn't what we want)
    max_time = dt.timedelta(hours=1.5) # how long each job is allowed to run before SLURM should kill it

    if max_time.days > 0:
        assert max_time.days == 1
        # 3600 = secs in 1 hour. 
        # max_time.seconds // 3600:02d = hours
        # (max_time.seconds % 3600) // 60:02d = minutes
        # max_time.seconds % 60:02d = seconds (because if mod 3600 = 0, then definitely mod 60 = 0, so can just mod 60 here (math works out, think abt it))
        # 02d = python formatter which says: format this number as an integer, padded with zeros to at least 2 digits wide
        max_time_str = f'1-{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'
    else:
        max_time_str = f'{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'

    data_dir = f'artifact/data/{cur_time.strftime("%Y%m%d_%H%M%S")}' # create a data directory (name contains current timestamp (y,m,d,h,m,s))
    # within data directory, we need: a config file, sbatch directory, output directory, log directory, benchmark directory, and metadata text file abt the job
    # I think it is one data directory per job array submitted????? Might very well be wrong tho TODO
    config_filename = f'{data_dir}/config.json' # create config file name within the data dir
    sbatch_dir = f'{data_dir}/sbatch' 
    output_dir = f'{data_dir}/output' 
    log_dir = f'{data_dir}/logs'
    benchmark_dir = f'{data_dir}/benchmarks'
    metadata_filename = f'{data_dir}/metadata.txt'
    # create actual directory folders
    os.makedirs(sbatch_dir)
    os.makedirs(output_dir)
    os.makedirs(log_dir)
    os.makedirs(benchmark_dir)

    # copy files to data dir to preserve them (the timestamp-specific data directory)
    shutil.copyfile('artifact/run_reaction_time_evals.py', f'{data_dir}/run_reaction_time_evals_copy.py') # left file = src, right file = destination/resulting copy of the file
    shutil.copyfile('artifact/_simulate.py', f'{data_dir}/_simulate_copy.py')

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    benchmark_names = {}
    memory_settings = None
    for file in os.listdir('artifact/data/cached_schedules/'):
        # USER SETTING: filter benchmark files if desired
        if file.endswith('.lss') and file.startswith('random_t_1000_200_0'): # only get the random cached schedules?
            path = os.path.join('artifact/data/cached_schedules/', file) # concatenates 2 file paths to create overall "big" path
            newpath = os.path.join(benchmark_dir, file) # add this benchmark test file into our benchmark directory in our final timestamped data directory
            # copy files to data dir to preserve them
            # specifically, copy the benchmark cached schedule test file from og place to the benchmark directory within our final timestamped data directory
            shutil.copyfile(path, newpath)
            benchmark_files.append(newpath)
            benchmark_names[newpath] = file.split('.')[0] # dict where key = path to new copied benchmark file, val = name of benchmark (without the .lss part)
        elif file == '.memory_settings.json':
            # USER SETTING: change values in this file to add memory for certain benchmarks
            memory_settings = json.load(open(os.path.join('artifact/data/cached_schedules/', file), 'r')) # load the memory settings json file, specifies per-benchmark memory requirements (in GB)
    assert memory_settings is not None

    for name in set(benchmark_names.values()): # iterate through all benchmark names (create a set to allow us to iterate through a UNIQUE set of benchmarks)
        if name not in memory_settings:
            memory_settings[name] = 4 # by default, if benchmark is not in memory settings json file, just give it 4 GB of memory for the requirement. 

    # USER SETTING: change parameter sweeps for distance, spec acc, etc.
    # basically, if we change this, we change the number of options we give each parameter that we iterate through for our testing
    # can see the details for many of these settings in decoder_manager.py
    sweep_params = {
        'benchmark_file':benchmark_files, # iterate through all benchmark files we want
        'distance':[21], # code distance
        'scheduling_method':['sliding', 'parallel', 'aligned'], # various scheduling methods for how we are decoding the windows (e.g. sliding vs parallel window decoding)
        'decoder_latency_or_dist_filename':[f'lambda volume: volume*{fac}' for fac in np.geomspace(0.1, 10, 10)], # TODO what is this? Something about string representations of lambda functions. geomspace creates 10 values btwn 0.1 and 10 inclusive that are spaced geometrically/multiplicatively
        'speculation_mode':['separate'], # different speculation modes (separate, integrated, None), see specs/details in decoder_manager.py
        'speculation_latency':[1], # speculation latency
        'speculation_accuracy':list(np.linspace(0, 1, 11)), # speculation accuracy, list of: 0, 0.1, 0.2, 0.3,...,0.9,1 (11 values from 0-1 inclusive)
        'poison_policy':['successors'], # dictates different number of windows that are restarted after a poisoned (incorrect) window/speculation. successors=reset only direct descendants that are directly dependent on speculation. descendants=restart all descendants
        'missed_speculation_modifier':[1.4], # factor by which incorrect speculation rate increases when an adjacent window is poisoned/has a missed speculation
        'max_parallel_processes':[None], # max # of parallel decoding processes that can be run at the same time
        'rng':[0],
        'lightweight_setting':[1],
    }
    ordered_param_names = list(sorted(sweep_params.keys())) # get all parameter names, and sort them alphabetically (so processed in consistent order)
    total_num_configs = reduce(lambda x,y: x*y, [len(params) for params in sweep_params.values()]) # this multiplies the length of every parameter's list. so, it gets the total number of possible parameter configs

    # Write config file (each Python job will read params from this)
    # write the config file for every possible combination of the configuration parameters
    configs = []
    cur_indices = [0 for _ in ordered_param_names] # array len = # parameter types (ex: benchmark_file, distance, scheudling_method, etc)
    for config_idx in range(total_num_configs): # current config_idx that we're on in total number of all possible configuration patterns we can have
        rolled_over = [False for _ in ordered_param_names] # array len = # parameter types
        config = {} # current config we want to create
        for i,name in enumerate(ordered_param_names): # i = idx of current param we're on, name = name of current param we're on
            idx = cur_indices[i] # the current index we're on in the list of the current parameter that we're on (idx we're on in the list of options for the parameter we're on)
            config[name] = sweep_params[name][idx] # get the current parameter name we're on (key) and the value for it from sweep_params and based on our current idx into the list for that param (value)
            # when i = 0, all() will return true because it's operating on an empty list
            # the incrementing for cur_idx will propagate starting from first param, to second param, and so on so forth
            # so the scan will be thru entire first param's list, then entire second param's list, and so on so forth (but will re-iterate through first list for new second param list idx)
            if all(rolled_over[:i]): # all returns True if every element in the array is True; else it returns False
                cur_indices[i] += 1
            if cur_indices[i] >= len(sweep_params[name]): # if we have iterated through the entire list for this parameter (current idx out of bounds)
                cur_indices[i] = 0 # set our current index back to 0
                rolled_over[i] = True # set that we have rolled over (so we're on another round for this parameter)
        configs.append(config)
    with open(config_filename, 'w') as f:
        json.dump(configs, f)

    # submit a different sbatch job for each config
    configs_by_mem = {}
    for i,config in enumerate(configs):
        mem_gb = memory_settings[benchmark_names[config['benchmark_file']]] # get the memory setting of the current benchmark
        # append the index of the current config to the appropriate list in configs_by_mem, where lists are separated by the amt of mem a config requires
        configs_by_mem.setdefault(mem_gb, []).append(i) # setdefault returns the value if it exists, else it appends the new key-value pair with default value of []

    last_submit_time = None
    job_ids = []
    submit_idx = 0
    for i,mem_gb in enumerate(sorted(configs_by_mem.keys())): # iterate through configs by memory json in increasing order of mem used
        config_indices = configs_by_mem[mem_gb]
        # number of jobs needed to finish running this list of configurations
        # max_tasks_per_job = # configurations 1 job array can handle
        # we do math.ceil, meaning we round up
        num_submissions = math.ceil(len(config_indices) / max_tasks_per_job) # caslake submission limit
        for j in range(num_submissions):
            if last_submit_time: # we want to sleep for just enough time so that btwn submissions, we have submission_delay # of seconds
                time.sleep(max(0, int((last_submit_time + submission_delay - dt.datetime.now()).total_seconds())))
            selected_config_indices = config_indices[j*max_tasks_per_job:(j+1)*max_tasks_per_job] # get the range of config indices that we want to run in this job array
            print(f'\tSubmitting {len(selected_config_indices)} / {len(configs)} jobs...')
            sbatch_filename = os.path.join(sbatch_dir, f'submit_{submit_idx}.sbatch') # sbatch filename, which is where we write the slurm job that we launch (the specific launch cmd is stored here)
            submit_idx += 1
            # TODO: I'm not sure if mem-per-cpu should be *1000? # OLD CODE: #SBATCH --mem-per-cpu={mem_gb*1000}
            with open(sbatch_filename, 'w') as f: # %a is replaced by array task id automatically
                f.write(f'''#!/bin/bash
#SBATCH --job-name={cur_time.strftime("%Y%m%d_%H%M%S")}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.out
#SBATCH --account={slurm_account}
#SBATCH --array={','.join([str(x) for x in selected_config_indices])}
#SBATCH --time={max_time_str}
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu={mem_gb}G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jl1543@princeton.edu


module purge
module load anaconda3/2024.6
eval "$(conda shell.bash hook)"
conda activate {path_to_python_env}
cd {path_to_swiper}

python -m artifact._simulate "{config_filename}" "{output_dir}" {int(max_time.total_seconds())}'''
                )

            if dry_run:
                print(f'\t\tDRY RUN: sbatch {sbatch_filename}')
            else:
                # this submits the slurm job by running the sbatch file (sbatch x.sbatch). Popen allows Python to run the SLURM file. Stdout and stderr capture SLURM's output so Python can read it
                p = subprocess.Popen(f'sbatch {sbatch_filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                lines = list(p.stdout.readlines()) # reads all lines of output from the sbatch file
                retval = p.wait() # wait for process to complete
                if retval != 0: # error occurred, print out error lines
                    print(lines)
                job_ids.append((int(lines[-1].decode('utf-8').rstrip().split(' ')[-1]), mem_gb, selected_config_indices)) # get the slurm job id number and append the tuple (job_id, mem_gb, selected_config_indices) to job_ids
                last_submit_time = dt.datetime.now() # marks when job is submitted
                if submission_delay.total_seconds() > 0: # if have submission delay, will print out job id of latest submitted job
                    print(f'\tSubmitted job {job_ids[-1][0]}')

    with open(metadata_filename, 'w') as f: # write all this information into the metadata text file
        f.write(f'Time: {cur_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Job IDs:\n')
        for i,(job_id, mem_gb, tasks) in enumerate(job_ids):
            f.write(f'    submit_{i}.sbatch: {job_id}. {mem_gb}GB RAM, configs {tasks}\n')
        f.write(f'Max clock time: {max_time_str}\n')
        f.write(f'Total num. tasks: {len(configs)}\n')
        f.write(f'Params:\n')
        for name,params in sweep_params.items():
            f.write(f'    {name}: {params}\n')