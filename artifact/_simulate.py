import sys, os
sys.path.append('.')
import json
import pickle
import datetime as dt
import numpy as np
import math
from swiper.simulator import DecodingSimulator
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule

if __name__ == '__main__':
    args = sys.argv

    config_filename = args[1]
    output_dir = args[2] # output directory
    max_job_time = dt.timedelta(seconds=int(args[3]))
    config_idx = int(os.environ['SLURM_ARRAY_TASK_ID']) # os.environ contains shell's environment variables. this gets the config idx value within the slurm job array

    start_time = dt.datetime.now()

    with open(config_filename, 'r') as f: # get the current config idx's config dict of parameters from the config file
        params = json.load(f)[config_idx]

    print(f'CONFIG {config_idx}')
    for key,val in params.items(): # print out the parameter and its corresponding value in this configuration
        print(f'{key}: {val}')

    assert len(params) == 12, 'Params list changed. Update this file!' # if there are now more different params that we want to test with, then we want to update this file
    # get all the parameter values for the current configuration
    benchmark_file = params['benchmark_file']
    distance = params['distance']
    scheduling_method = params['scheduling_method']
    decoder_latency_or_dist_filename = params['decoder_latency_or_dist_filename']
    speculation_mode = params['speculation_mode']
    speculation_accuracy = params['speculation_accuracy']
    speculation_latency = params['speculation_latency']
    poison_policy = params['poison_policy']
    missed_speculation_modifier = params['missed_speculation_modifier']
    max_parallel_processes = params['max_parallel_processes']
    lightweight_setting = params['lightweight_setting']

    rng = params['rng']

    generator = np.random.default_rng(rng)

    # decoder_dists[str(d)][str(volume)] = 10,000 sampled latencies, in microseconds
    # TODO I still don't know what's in the decoder dict file?
    # This migh tbe returning the group of possible decoder latencies that corresopnd with a particular volume/distance
    # 3 options for how decoder latency is given -- as a file(json), as a string directly of code, or as simply an int
    if isinstance(decoder_latency_or_dist_filename, str) and decoder_latency_or_dist_filename.endswith('.json'): # get the json file with all the decoder latency parameters
        decoder_dists = json.load(open(decoder_latency_or_dist_filename, 'r'))
        decoder_dist = {}
        for dist_str, dist_dict in decoder_dists.items(): # iterate throuhg all decoder distance options
            if int(dist_str) == distance: # if the current decoder distance that we're looking at is the same as the configuration's decoder distance
                decoder_dist = {int(k):v for k,v in dist_dict.items()} # take this decoder distance's dict (but convert its keys to ints)
        # this might be picking one decoder latency for the corresponding decoder distance/volume that we're currently on? TODO not sure tho
        decoding_latency_fn = lambda volume: generator.choice(decoder_dist[max(2, math.ceil(volume / distance))]) # anonymous function with one argument volume, generator.choice randomly picks one value from list
    elif isinstance(decoder_latency_or_dist_filename, str):
        decoding_latency_fn = eval(decoder_latency_or_dist_filename) # evaluate this string as python code in runtime, executes it as if it were written in the pgm
    else:
        decoding_latency = int(decoder_latency_or_dist_filename)
        decoding_latency_fn = lambda _: decoding_latency

    print(f'{start_time.strftime("%Y-%m-%d %H:%M:%S")} | Loading benchmark {benchmark_file}...') # print current benchmark file we're on, print it out directly to stdout immediately
    sys.stdout.flush()

    with open(benchmark_file, 'r') as f:
        benchmark_schedule = LatticeSurgerySchedule.from_str(f.read(), generate_dag_incrementally=True) # create a lattice surgery schedule for this benchmark

    # run simulator with current configuration's settings
    simulator = DecodingSimulator()
    success, simulator_params, device_data, window_data, decoding_data = simulator.run(
        schedule=benchmark_schedule,
        distance=distance,
        scheduling_method=scheduling_method,
        decoding_latency_fn=decoding_latency_fn,
        speculation_mode=speculation_mode,
        speculation_latency=speculation_latency,
        speculation_accuracy=speculation_accuracy,
        poison_policy=poison_policy,
        missed_speculation_modifier=missed_speculation_modifier,
        max_parallel_processes=max_parallel_processes,
        print_interval=dt.timedelta(seconds=10),
        lightweight_setting=lightweight_setting,
        clock_timeout = max_job_time - dt.timedelta(minutes=5), # allow 5 mins for starting + finishing job
        rng=rng,
    )

    # dump all results from simulator into an output file
    simulator_params_dict = simulator_params.to_dict()
    simulator_params_dict['decoding_latency_fn'] = decoder_latency_or_dist_filename
    with open(os.path.join(output_dir, f'config{config_idx}_d{distance}_{scheduling_method}_{speculation_mode}_{benchmark_file.split("/")[-1].split(".")[0]}_{rng}.json'), 'w') as f:
        json.dump({
                'success':success,
                'simulator_params':simulator_params_dict,
                'device_data':device_data.to_dict(),
                'window_data':window_data.to_dict(),
                'decoding_data':decoding_data.to_dict(),
            },
            f
        )

    # bookeeping print for total time taken
    finish_time = dt.datetime.now()
    print(f'{finish_time.strftime("%Y-%m-%d %H:%M:%S")} | Finished saving output. Done! Total elapsed time: {finish_time - start_time}')
    sys.stdout.flush()