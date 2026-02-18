"""This script evaluates various strategies for handling a misprediction
(varying the number of downstream decode tasks that we reset upon realizing that
a prior dependency was decoded incorrectly). It saves the results to a file for
later analysis.

This script takes under five minutes to run on an M2 Macbook Pro."""

import os, sys
sys.path.append('.')
import json
import networkx as nx
from swiper.predictor import strategy_sim

if __name__ == '__main__':
    d=13 # code distance
    num_nodes = 100 # number of "stabilizer nodes" that we have / (number of windows we're decoding (length of the sequence of windows that we're decoding))
    num_shots = 10_000 # number of shots taken = # of full measurement rounds where every syndrome bit is measured
    decode_times = [2 * 13, 5 * 13, 10 * 13] # list of all decoding latencies to test (how long does it take to fully decode a window). each cycle takes time d (d=13 here) ("new window arrives every d rounds"), and we test it for decoding latencies of cycle length 2, 5, 10
    test_graph = nx.DiGraph() # create empty directed graph
    test_graph.add_nodes_from([(node, {'t': d * node}) for node in range(num_nodes)]) # add all nodes. each node gets a time attribute/timestamp 't' (so node 0 gets t = 0, node 1 t = 13, node 2 t = 26, so on)
    test_graph.add_edges_from([(i, i+1) for i in range(num_nodes - 1)]) # add edges to connect nodes together by: 0 -> 1 -> 2 -> 3 -> ... -> num_nodes-1
    adj_pairs = list(zip(list(test_graph.edges())[:-1], list(test_graph.edges())[1:])) # get all adjacent pairs of edges --> ex: [((0,1),(1,2)), ((1,2),(2,3)), …]. [:-1] and [1:] ofsets them by 1, and zip pairs them up to form the adjacent pairs of edges.
    # for every misprediction strategy, create a list of runtimes, classical workload times, and peak parallelism times (peak number of decoders running in parallel at the same time), for every decode time
    opt_runtimes = {decode_time: [] for decode_time in decode_times}
    opt_classicals = {decode_time: [] for decode_time in decode_times}
    opt_procs = {decode_time: [] for decode_time in decode_times}
    pes_runtimes = {decode_time: [] for decode_time in decode_times}
    pes_classicals = {decode_time: [] for decode_time in decode_times}
    pes_procs = {decode_time: [] for decode_time in decode_times}
    adj_runtimes = {decode_time: [] for decode_time in decode_times}
    adj_classicals = {decode_time: [] for decode_time in decode_times}
    adj_procs = {decode_time: [] for decode_time in decode_times}

    print(f"Running misprediction strategy simulation for d = {d}, num_nodes = {num_nodes}, num_shots = {num_shots}, decode_times = {decode_times}")
    for decode_time in decode_times:
        print(f"Running decode_time = {decode_time}. Progress: ", end='', flush=True)
        for shot in range(num_shots): # iterate through all shots
            if shot % (num_shots // 10) == 0:
                print(f"{shot / num_shots * 100: 0.0f}% ", end='', flush=True)
            # pass in directed graph that was created with nodes (windows) and edges, pass in adjacent edges pairs, pass in current decode time/latency (see rest of meanings in method itself)
            opt_runtime, opt_classical, opt_valid, opt_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'optimistic')
            assert opt_valid == num_nodes * decode_time # we expect the valid amount of classical time to simply be the number of nodes multiplied by the amount of time it takes to decode each one (this is without speculation, just base time)
            pes_runtime, pes_classical, pes_valid, pes_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'pessimistic')
            assert pes_valid == num_nodes * decode_time
            adj_runtime, adj_classical, adj_valid, adj_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'adjacent')
            assert adj_valid == num_nodes * decode_time
            # for our current decode time, append our current results to the appropriate list for the current shot that we're on
            opt_runtimes[decode_time].append(opt_runtime)
            opt_classicals[decode_time].append(opt_classical)
            opt_procs[decode_time].append(opt_proc)
            pes_runtimes[decode_time].append(pes_runtime)
            pes_classicals[decode_time].append(pes_classical)
            pes_procs[decode_time].append(pes_proc)
            adj_runtimes[decode_time].append(adj_runtime)
            adj_classicals[decode_time].append(adj_classical)
            adj_procs[decode_time].append(adj_proc)
        print()
    
    # dump all our results into a json file
    with open('artifact/data/mispredict_data.json', 'w') as f:
        json.dump({
            'd': d,
            'num_nodes': num_nodes,
            'num_shots': num_shots,
            'decode_times': decode_times,
            'opt_runtimes': opt_runtimes,
            'opt_classicals': opt_classicals,
            'opt_procs': opt_procs,
            'pes_runtimes': pes_runtimes,
            'pes_classicals': pes_classicals,
            'pes_procs': pes_procs,
            'adj_runtimes': adj_runtimes,
            'adj_classicals': adj_classicals,
            'adj_procs': adj_procs,
        }, f)