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

In our experiments, we ran two separate instances of this script with different
parameters, running most benchmarks on a large cluster with a 36-hour job time
limit and running a few larger benchmarks on a smaller cluster with a 7-day job
time limit. Here, we combine them into a single script that runs all benchmarks 
with a 7-day job time limit. See `data/benchmarks1/submit_jobs_copy.py`
and `data/benchmarks2/submit_jobs_copy.py` for the
original job submission scripts used in the paper.

This script is adapted from `slurm/run_simulation.py` and `slurm/submit_jobs.py`
in the original repository.

This script can take between 1 minute and several hours to submit all jobs,
depending on configuration of `submission_delay` and `max_tasks_per_job`. The
SLURM jobs themselves will take several days to complete."""

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
import csv

################################################################################
# CONFIGURATION
################################################################################

dry_run = False # if True, don't actually submit jobs, just print sbatch commands

# Required configuration
slurm_account = 'martonosi' # argument to `--account` in `sbatch`
slurm_partition = 'n/a?' # argument to `--partition` in `sbatch`
path_to_python_env = 'swiper' # argument to `conda activate` in sbatch script
path_to_swiper = '/home/jl1543/swiper' # path to the root of this repository

# Optional configuration
submission_delay = dt.timedelta(minutes=30) # too long delay -- TODO might change it down to only 1 minute at most, if even 0 minutes
max_tasks_per_job = 500
min_circuit_t_count = None
max_circuit_t_count = None
# for the decoder_latency_or_dist_filename parameter
decoder_dist_source = 'artifact/data/decoder_dists.json' # can change if this has been re-generated elsewhere by running `artifact/run_pymatching_latencies.py`

################################################################################
# makes sure that options are configured
if not dry_run and any([x == 'CHANGE_ME' for x in [slurm_account, slurm_partition, path_to_python_env, path_to_swiper]]):
    raise ValueError('Please configure the variables at the top of this script.')

if __name__ == '__main__':
    cur_time = dt.datetime.now()

    # USER SETTING: maximum job duration
    max_time = dt.timedelta(hours=80) # set max job duration to be 1 week (24*7 hours) -- but I changed this to only 80 hours since della has an upper limit on max time length

    if max_time.days > 0:
        # in format of days-hrs:mins:seconds
        max_time_str = f'{max_time.days}-{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'
    else:
        # in format of hrs:mins:seconds
        max_time_str = f'{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'

    data_dir = f'artifact/data/{cur_time.strftime("%Y%m%d_%H%M%S")}'
    # create directories within the data directory generated from the test
    config_filename = f'{data_dir}/config.json'
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

    # copy files to data dir to preserve them
    # copy decoder_dists_json file too, to the location {data_dir}/decoder_dists.json
    decoder_dist_filename = f'{data_dir}/{decoder_dist_source.split("/")[-1]}'
    shutil.copyfile(decoder_dist_source, decoder_dist_filename)
    shutil.copyfile('artifact/run_benchmark_evals.py', f'{data_dir}/run_benchmark_evals_copy.py')
    shutil.copyfile('artifact/_simulate.py', f'{data_dir}/_simulate_copy.py')

    # this csv contains all benchmark information. 
    with open('artifact/data/benchmark_info.csv', 'r') as f:
        reader = csv.DictReader(f) # Read csv file, and return it as a dictionary, where key=header, value=value of that header for a specific row
        benchmark_info = {row['']:row for row in reader} # Creates a wrapper dict, with key='', value=a row from the csv which is in a dict format

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    benchmark_names = {}
    memory_settings = None
    for file in os.listdir('artifact/data/cached_schedules/'):
        # USER SETTING: filter benchmark files if desired
        # get all benchmark file schedules
        if file.endswith('.lss') and not file.startswith('memory') and not file.startswith('regular') and not file.startswith('random'):
            path = os.path.join('artifact/data/cached_schedules/', file)
            newpath = os.path.join(benchmark_dir, file) # new path to put benchmark in resulting folder
            # copy files to data dir to preserve them
            shutil.copyfile(path, newpath)
            benchmark_files.append(newpath)
            benchmark_names[newpath] = file.split('.')[0] # newpath of benchmark = key, actual benchmark name=value
        elif file == '.memory_settings.json':
            # USER SETTING: change values in this file to add memory for certain benchmarks
            memory_settings = json.load(open(os.path.join('artifact/data/cached_schedules/', file), 'r'))
    assert memory_settings is not None

    for name in set(benchmark_names.values()): # iterate through all benchmark names, check if their memory setting is in memory_setting_json; if not, add it with default 4 GB memory
        if name not in memory_settings:
            memory_settings[name] = 4

    # USER SETTING: change parameter sweeps for distance, spec acc, etc.
    sweep_params = {
        'benchmark_file':benchmark_files, # iterate through all benchmark file paths
        'distance':[15, 21, 27], # code distance
        'scheduling_method':['sliding', 'parallel', 'aligned'], # various scheduling methods for how we are decoding the windows (e.g. sliding vs parallel window decoding)
        'decoder_latency_or_dist_filename':[decoder_dist_filename], # needs to be a list of decoder distances?? TODO not quite sure what this is
        'speculation_mode':['separate', None], # different speculation modes (separate, integrated, None), see specs/details in decoder_manager.py
        'speculation_latency':[1], # speculation latency
        'speculation_accuracy':[0.9, 1.0], # speculation accuracy, list of: 0, 0.1, 0.2, 0.3,...,0.9,1 (11 values from 0-1 inclusive)
        'poison_policy':['successors'], # dictates different number of windows that are restarted after a poisoned (incorrect) window/speculation. successors=reset only direct descendants that are directly dependent on speculation. descendants=restart all descendants
        'missed_speculation_modifier':[1.4], # factor by which incorrect speculation rate increases when an adjacent window is poisoned/has a missed speculation
        'max_parallel_processes':[None, 'predict'], # predict is probably the number of max # of parallel processors that the test is expected to use. | # max # of parallel decoding processes that can be run at the same time
        'rng':list(range(10)),
        'lightweight_setting':[2],
    }
    ordered_param_names = list(sorted(sweep_params.keys())) # sort parameters alphabetically, get list of all names
    total_num_configs = reduce(lambda x,y: x*y, [len(params) for params in sweep_params.values()])

    # USER SETTING: filter out some combinations of the above parameters
    microbenchmarks = [os.path.join(benchmark_dir, file) for file in ['msd_15to1.lss', 'adder_n4.lss', 'adder_n10.lss', 'adder_n18.lss', 'adder_n28.lss' 'rz_1e-05.lss', 'rz_1e-10.lss', 'rz_1e-15.lss', 'rz_1e-20.lss', 'toffoli.lss', 'qrom_15_15.lss']]
    def config_filter(cfg):
        # TODO: make the logic here more clear...
        return (
            (cfg['distance'] == 21 or (cfg['speculation_accuracy'] == 0.9 and cfg['max_parallel_processes'] == None)) # distance 15 and 27 runs require less data
            and (not (cfg['speculation_accuracy'] == 1.0 and (cfg['speculation_mode'] == None or cfg['max_parallel_processes'] == 'predict')))
            and (not (cfg['speculation_mode'] == None and (cfg['scheduling_method'] == 'sliding' or cfg['max_parallel_processes'] == 'predict'))) # don't want to turn off swiper for sliding window, or for predicting computational cost
            and (min_circuit_t_count is None or (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) > min_circuit_t_count)) # I presume t count has to do with smth abt t gates? 
            and (max_circuit_t_count is None or (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) < max_circuit_t_count)) 
            and (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) < 3500 or cfg['rng'] == 0) # only small benchmarks get multiple runs
        )

    # Write config file (each Python job will read params from this)
    configs = []
    cur_indices = [0 for _ in ordered_param_names]
    # go through every possible config combination of parameters, and create all possible configs
    for config_idx in range(total_num_configs):
        rolled_over = [False for _ in ordered_param_names]
        config = {}
        for i,name in enumerate(ordered_param_names):
            idx = cur_indices[i]
            config[name] = sweep_params[name][idx]
            if all(rolled_over[:i]):
                cur_indices[i] += 1
            if cur_indices[i] >= len(sweep_params[name]):
                cur_indices[i] = 0
                rolled_over[i] = True
        if config_filter(config): # only include configs that match the filter
            configs.append(config)
    print('Generated', len(configs), 'configs')
    with open(config_filename, 'w') as f:
        json.dump(configs, f)

    # submit a different sbatch job for each config
    configs_by_mem = {}
    for i,config in enumerate(configs):
        mem_gb = memory_settings[benchmark_names[config['benchmark_file']]] # lookup the memory setting that this specific benchmark config takes
        configs_by_mem.setdefault(mem_gb, []).append(i) # group configs by memory. key = memory in gb, value = list of configs with that memory requirement

    last_submit_time = None
    job_ids = []
    submit_idx = 0
    for i,mem_gb in enumerate(sorted(configs_by_mem.keys())):
        config_indices = configs_by_mem[mem_gb] # get the config indices (val of the dict)
        # get the number of submissions we need to make (number of job arrays that we have to submit, since each job array has a maximum length)
        num_submissions = math.ceil(len(config_indices) / max_tasks_per_job) # caslake submission limit
        for j in range(num_submissions):
            if last_submit_time:
                time.sleep(max(0, int((last_submit_time + submission_delay - dt.datetime.now()).total_seconds()))) # make sure that there is submission_delay time btwn two consecutive submissions
            selected_config_indices = config_indices[j*max_tasks_per_job:(j+1)*max_tasks_per_job] # get the group of config indices that we want to submit in this job
            print(f'\tSubmitting {len(selected_config_indices)} / {len(configs)} jobs...')
            sbatch_filename = os.path.join(sbatch_dir, f'submit_{submit_idx}.sbatch') # make the sbatch file that we want to run/submit
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