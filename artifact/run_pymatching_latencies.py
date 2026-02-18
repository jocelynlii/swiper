"""This script collects PyMatching latencies for randomly generated surface code
decoding problems of varying distance and number of rounds, then saves them to a
file for use in the benchmark simulations.

On an M1 Macbook Pro, this script takes about 1 hour to run."""

import os, sys
sys.path.append('.')
import stim
import pymatching
import numpy as np
import datetime
import json
import pickle as pkl

if __name__ == '__main__':
    num_shots = 10_000
    d_range = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31] # all code distances we want to collect latencies over
    decoding_dists = {d: {} for d in d_range}
    p = 1e-3 # error probability

    print(f"Running pymatching decoder latencies for d in {d_range} and p = {p}")
    for d in d_range:
        print(f"Running d = {d}, r = ", end='', flush=True)
        for r in range(2, 8): # r = different coefficients for window volumes (2,...,7)
            print(f"{r} ", end='', flush=True)
            circ = stim.Circuit.generated("surface_code:rotated_memory_z", # type of prebuilt stim circuit (memory experiment, no logical gates, focus measure logical-Z logical error rate)
                                    distance=d, rounds=r*d, # rounds = number syndrome measurement rounds. Space-time volume/total number of measurements = distance * rounds. Space = d^2, rounds/rounds of measurement (measure every qubit once) = r*d. 
                                    after_clifford_depolarization=p, # introduce Pauli depolarizing noise after each Clifford gate, to model gate errors during stabilizer measurements (adds noise)
                                    before_measure_flip_probability=p, # probability p that measurement operation result is flipped. Models measurement errors (adds noise)
                                    )
            # detector error model maps which detectors will click if certain events happen (basically models the map that deduces what type of error we have).
            # builds a min weight perfect matching graph based on this DEM. Used to infer likely error chains from observed syndrome data
            matching = pymatching.Matching.from_detector_error_model(circ.detector_error_model()) 
            # Generates synthetic detector outcomes for each full run of circuit (shot). Outputs array of 0/1 flags for every syndrome bit's measurement result.
            sampler = circ.compile_detector_sampler()
            # shots = 2d array of size (num_shots, num_detectors) of every syndrome bit measurement result. 
            # actual_observables = ground-truth logical observables (logical Z error flipped/didn't flip the logical qubit)
            shots, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
            # Decode one shot first to ensure internal C++ representation of the matching graph is fully cached
            matching.decode_batch(shots[0:1, :]) # get all the syndrome bit measurement results of just one shot
            decoding_dists[d][r] = np.zeros(num_shots) # for every volume, initialize array for every shot taken
            # Now time decoding the batch
            for i in range(num_shots):
                shot = shots[i:i+1, :] # every shot is an entire row of the shots array (each row represents all syndrome bit measurements in 1 shot)
                t0 = datetime.datetime.now()
                matching.decode(shot) # matching is a DEM/graph that decodes/infers the most likely error chain from the observed syndrome data
                t1 = (datetime.datetime.now() - t0).total_seconds() * 1e6 # us
                decoding_dists[d][r][i] = t1 # i = shot number that we're on. t1 = time it took to decode this specific shot with the specific configurations. r = window decoding volume coefficient. d = code distance.
        print()
    
    decoding_dists_listified = {d: {r: decoding_dists[d][r].tolist() for r in decoding_dists[d]} for d in decoding_dists} # convert numpy array to an actual list
    pkl.dump(decoding_dists_listified, open('artifact/data/decoder_dists.pkl', 'wb')) # wb = file opened for binary read. .pkl file created here
    with open('artifact/data/decoder_dists.json', 'w') as f: # w = file opened for write
        json.dump(decoding_dists_listified, f) # converts list into a json, and dumps it into the file; .json file created here