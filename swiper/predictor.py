import stim
import pymatching
import numpy as np
import pickle as pkl
import datetime
import networkx as nx

# takes in the samples of all the shots (2d array of ints, each row is a shot, where every element is a measurement of a syndrome bit)
# weight_1_chains are edges that are on the temporal boundary (lead to dependencies)
# weight_2_chains are len 2 edges that cross the boundary (commit --> buffer)
# ignore_steps are for if u only want to do 1 or 2-step predictor
# one_step_chains are len 1 chains who cross btwn a temporal boundary (exactly go from commit --> buffer)
def speculator(samples, weight_1_chains, weight_2_chains, ignore_step=[], one_step_chains=[]):
    shared_syndromes = samples
    output_syndromes = shared_syndromes.copy() # need a copy bc shared_syndromes will be used for enumerating so don't want to edit, while output_syndromes is for us to edit to get the output
    believed_match = [[False for _ in range(len(weight_1_chains))] for _ in range(len(samples))] # only look at weight 1 chains here? dim: # shots x weight_1_chains len
    matchings = [[] for _ in range(len(samples))] # list of matchings for each shot (len(samples) = # shots taken)
    def step1(i, chain, match=False): # i = idx of chain were looking at; chain = actual chain/edge that we're looking at; match = if true, automatically record chain as predicted match
        for j, shared_sample in enumerate(shared_syndromes): # iterate through all samples (entire measurement round of all syndrome bits of 1 shot; shared_sample is list that stores results)
            if len(chain) == 1: # chain is the tuple of endpoints that we store. so, len(chain) == 1 only if one of our endpoints is the boundary node (these are never in one_step_chain list). len(chain) == 2 if this is not the case (both nodes are just within the temporal boundary)
                # if in the sample, the endpoint/decoder bit that's not a boundary node is measured as 1, then for our current shot and our current decoder bit, add 50 to its "score" in output_syndromes
                if shared_sample[chain[0]]: 
                    output_syndromes[j][chain[0]] += 50
                    believed_match[j][i] = True # i = idx of chain we're looking at in one_step_chains/weight_1_chains. j = shot #. len(one_step_chains)<=len(weight_1_chains) by how it's constructed
            else: # case where both our endpoints are in temporal boundary 
                if shared_sample[chain[0]] and shared_sample[chain[1]]: # if both endpoints (syndrome bits) measure as 1
                    # increment both syndrome bits' scores by 1, for the 2-step and 3-step predictor
                    output_syndromes[j][chain[0]] += 1 # bump "score" on output_syndromes array at the endpoint/syndrome bit ID location
                    output_syndromes[j][chain[1]] += 1
                    believed_match[j][i] = True
                    if match: # in 1-step predictor, directly predict that error occurred and create a dependency. So, add this edge/chain to the matchings array
                        matchings[j].append(chain)
        return
    def step2(i, j, chain): # j = shot number, chain = current chain that we're looking at
        if len(chain) == 1: # if look at an edge with boundary node as one of its endpoints
            if output_syndromes[j][chain[0]] > 0: # if its one temporal boundary endpoint is more than 0, add to speculated matchings
                output_syndromes[j][chain[0]] = 0
                matchings[j].append(chain)
        else: # if look at edge with 2 temporal boundary nodes as endopints
            if output_syndromes[j][chain[0]] > 0 and output_syndromes[j][chain[1]] > 0: # if both endpoints > 0, add to speculated matchings
                output_syndromes[j][chain[0]] = 0
                output_syndromes[j][chain[1]] = 0
                matchings[j].append(chain)
    def step3(i, chain): # 
        for j, shared_sample in enumerate(shared_syndromes): # look at all syndrome bit scores for one shot
            if shared_sample[chain[0]] and shared_sample[chain[1]]: # if both are 1 in the weight 2 chain, add to matching
                matchings[j].append(chain)
        return
    
    if 1 not in ignore_step: # not ignore step 1
        if [2,3] == ignore_step:
            # for 1-step predictor, only look at single errors that cross boundary btwn commit and buffer region --> these only satisfied by one_step_chains
            for i, weight_1_chain in enumerate(one_step_chains):
                step1(i, weight_1_chain, match=True)
        else: # for any other predictor, look at all weight 1 chains with step 1. note match=false here
            for i, weight_1_chain in enumerate(weight_1_chains):
                step1(i, weight_1_chain)

        # create the "score" arrays for all of the nodes, "score" arrays produced by step 1
        shared_syndromes = output_syndromes
        output_syndromes = shared_syndromes.copy()
    if 2 not in ignore_step:
        for j, shared_sample in enumerate(shared_syndromes): # go through every shot's row of scores for every node currently, after step-1 changed scores
            # sort indices i of weight_1_chains. for the current shot and looking at all scores of all nodes, sort by ascending order, where score of temporal_boundary-temporal_boundary edges is sum of 2 nodes' scores, and score of boundary node edges is just 100. so boundary singletons usually last in this sorted order
            sorted_1_indices = sorted(range(len(weight_1_chains)), key=lambda i: shared_sample[weight_1_chains[i][0]] + (shared_sample[weight_1_chains[i][1]] if len(weight_1_chains[i]) > 1 else 100))
            for i in sorted_1_indices:
                step2(i, j, weight_1_chains[i]) # i = index of current chain we're looking at in weight_1_chains, j = current shot we're looking at, weight_1_chains[i] = actual chain we're looking at
        # update the score matrices once again (in particular the zero scores)
        shared_syndromes = output_syndromes
        output_syndromes = shared_syndromes.copy()
    if 3 not in ignore_step:
        # for all chains that cross the boundary with weight 2, we run step3 on it
        for i, weight_2_chain in enumerate(weight_2_chains):
            step3(i, weight_2_chain)

    return matchings # list of length #shots. for every element, it's a list of chains that it we speculate occurred in that shot. From these chains, the ones that cross the commit-buffer cut are the ones that create dependency bits to pass to the next window
    
# Extract data dependencies given the syndrome bits, the matching of error chains, the coordinates dictionary (maps detector id to geometric location in 3D decoding lattice), 
# boundary_idx (list of IDs near the boundary), and boundary_ub (marks split between commit and buffer region)
def extract_data_deps(syndrome, matching, coords_dict, boundary_idx, boundary_ub, nx_G, boundary_node=-1):
    dep_matchings = {}
    output_syndrome = syndrome.copy()
    boundary_dets = [det for det, coords in coords_dict.items() if coords[boundary_idx] == boundary_ub] # get all the detectors that are actually ON the boundary
    det_lookup = {tuple(coords): det for det, coords in coords_dict.items()} # dictionary used to lookup for a specific coordinate, what detector id it corresponds to
    
    # for each chain, a coord is -1 if it is the special boundary node
    for chain in matching:
        boundary_match = None
        if len(chain) == 1: # if chain tuple is of length 1, then we know it's a boundary match
            boundary_match = chain[0]
        elif chain[0] == -1: 
            boundary_match = chain[1]
        elif chain[1] == -1:
            boundary_match = chain[0] 
        if boundary_match: # ignore all of these special boundary match cases
            continue
        det1, det2 = chain
        if det1 == -1 or det2 == -1: # extra check to ensure none of the nodes are a special coord of -1
            continue
        # get 2 endpoint detector's time coordinate (which, within the coordinate tuple, is idx 2 (boundary_idx gives this idx 2))
        t_coord1 = coords_dict[det1][boundary_idx]
        t_coord2 = coords_dict[det2][boundary_idx] 
        min_t, max_t = min(t_coord1, t_coord2), max(t_coord1, t_coord2) # get the minimum and maximum time
        if min_t < boundary_ub and max_t >= boundary_ub: # if one detector's timestamp is in commit rgn, other detector's timestamp is in buffer rgn (aka, if edge crosses the boundary)
            # store this detector matching into dep_matchings --> key = detector with smaller timestamp, val = larger timestamp detector
            if t_coord1 < t_coord2:
                dep_matchings[det1] = det2
            else:
                dep_matchings[det2] = det1

            # Create an artificial syndrome bit/boundary detector at this boundary/crossing
            if t_coord2 == max_t:
                # det_lookup will lookup the detector Id for a specific coordinate. We want to get the detector Id of this artificial detector
                # For the coordinate tuple that we're looking up, we want the spatial coordinates of the current detector, then the time coordinate of the boundary, and then any remaining dimensions from the detector
                artificial_det = det_lookup[tuple(coords_dict[det2][:boundary_idx]) + (boundary_ub,) + tuple(coords_dict[det2][boundary_idx+1:])]
                output_syndrome[artificial_det] ^= True # now in the output syndrome data, we insert this artificial detector bit into it by XOR'ing (flipping) it 
            elif t_coord1 == max_t:
                artificial_det = det_lookup[tuple(coords_dict[det1][:boundary_idx]) + (boundary_ub,) + tuple(coords_dict[det1][boundary_idx+1:])]
                output_syndrome[artificial_det] ^= True
                
    return [output_syndrome[det] for det in boundary_dets], dep_matchings # return the boundary's dependency bits, along with the corresponding error chains that lead to these dependency bits

# Verify whether the speculated and real data dependencies are the same
def verify_speculation(syndrome, spec_matches, real_matches, coords_dict, boundary_idx, boundary_ub, nx_G):
    spec_syndrome, spec_deps  = extract_data_deps(syndrome, spec_matches, coords_dict, boundary_idx, boundary_ub, nx_G, len(coords_dict))
    real_syndrome, real_deps  = extract_data_deps(syndrome, real_matches, coords_dict, boundary_idx, boundary_ub, nx_G)
    return spec_syndrome == real_syndrome, spec_syndrome, real_syndrome

def add_noise(circ, p):
    def transform(block: stim.Circuit):
        noisy = stim.Circuit()

        for inst in block:
            if inst.name == "REPEAT":
                body = inst.body_copy()
                noisy_body = transform(body)
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue

            noisy.append(inst)

            # depolarizing after 2-qubit gates
            if inst.name == "CX" or inst.name == "CZ":
                noisy.append("DEPOLARIZE2", inst.targets_copy(), p)

            # measurement noise
            elif inst.name in ("M", "MX", "MY"):
                noisy.append("X_ERROR", inst.targets_copy(), p)
                # noisy.append("Z_ERROR", inst.targets_copy(), p)

        return noisy
    
    return transform(circ)

# num_shots = number of shots taken, d = code distance, p = physical error probability, ignore_steps = ignore some step of the predictor
def simulate_temporal_speculation(num_shots, d, p, ignore_steps=[]):
    # circ = stim.Circuit.generated("surface_code:rotated_memory_z", # type of prebuilt stim circuit
    #                           distance=d, rounds=2*d, # distance = code distance, rounds = # measurement rounds. total volume/# measurements = 2d^3
    #                           after_clifford_depolarization=p, # pauli depolarizing noise after each clifford gate (error)
    #                           before_measure_flip_probability=p, # measurement outcome flipped probability (error)
    #                           )
    # print(circ)
    circ = stim.Circuit.from_file("y_meas.stim")
    # circ = add_noise(circ, p)
    # print(circ)
    coords_dict = circ.detector_error_model().get_detector_coordinates() # map detector IDs to detector coordinates in the decoder graph. Coords = (x,y,t)
    print(len(coords_dict))
    dem = circ.detector_error_model()
    print(any(line.startswith("error(") for line in str(dem).splitlines()))
    matching = pymatching.Matching.from_detector_error_model(circ.detector_error_model()) # matching object built from Stim's detector error model (like a min weight decoding matching map used to figure out what error chain we're looking at)
    boundary_node = len(coords_dict) # the last detector (reserved as special boundary node)

    sampler = circ.compile_detector_sampler() # create sampler to get all the syndrome bit measurements
    # Mark all coordinates whose time coordinates lie near the window boundary. These are the detectors relevant for temporal data dependencies btwn windows
    temporal_boundary_mask = [False for _ in range(len(coords_dict))]
    for det, coords in coords_dict.items():
        if coords[2] >= d - 2 and coords[2] <= d + 2:
            temporal_boundary_mask[det] = True
    temporal_boundary_mask = np.array(temporal_boundary_mask)

    nx_G = matching.to_networkx() # nodes = detectors, edges = possible error events that flip one or more detectors. Convert to NetworkX object
    edges_to_remove = []
    nodes_to_remove = np.where(temporal_boundary_mask == False)[0] # remove all nodes not within the temporal boundary range ([0] grabs first element of tuple (tuple for multi-dimension, but here only 1D))
    # In this for loop, we want to remove all edges who are not fully within some boundary (temporal or spatial)
    for edge in nx_G.edges(): # edge[0], edge[1] are 2 endopints of the edge
        if edge[0] == boundary_node: # one endpoint is the boundary node
            if not temporal_boundary_mask[edge[1]]: # other endpoint is not in the temporal boundary range mask
                edges_to_remove.append(edge) # remove edge. So that boundary node only stays connected to boundary detectors
            continue
        elif edge[1] == boundary_node: # flipped version of previous if-clause
            if not temporal_boundary_mask[edge[0]]:
                edges_to_remove.append(edge)
            continue
        elif not temporal_boundary_mask[edge[0]] or not temporal_boundary_mask[edge[1]]: # if any endpoint not in temporal boundary mask, remove (only keep edges fully in temporal boundary region)
            edges_to_remove.append(edge)
    # After this pruning, the graph only contains teh boundary node and the temporal boundary detectors, and edges only connect these nodes
    nx_G.remove_edges_from(edges_to_remove)
    nx_G.remove_nodes_from(nodes_to_remove)
            
    det_dists = dict(nx.all_pairs_shortest_path_length(nx_G)) # computes shortest path distances between all pairs of nodes in the reduced decoder graph
    # print(det_dists)
    def get_dep_weight(det1, det2): # get weight of the dependency (min # of single-qubit errors/edges to connect the 2 nodes/detectors)
        # return det_dists[det1][det2]
        return det_dists.get(det1, {}).get(det2)

    weight_1_chains = [] # single edge matchings btwn boundary detectors. All chains of weight 1 that touch the temporal boundary region
    one_step_chains = [] # special case of weight-1 chains crossing btwn rounds d-1 and d (cross temporally)
    for det1, det2 in nx_G.edges(): # det1 and det2 are the 2 endpoints of the edge
        if det1 == boundary_node: # one endpoint is the boundary node
            if temporal_boundary_mask[det2]: # other endpoint is in the temporal boundary band
                weight_1_chains.append((det2,)) # touch temporal boundary + is weight 1, since the other endpt is the boundary node
            continue
        if det2 == boundary_node: # vice versa of previous if clause
            if temporal_boundary_mask[det1]:
                weight_1_chains.append((det1,)) # the tuple with only 1 element indicates that the other side is a boundary node
            continue
        # Edge therefore weight-1 chain
        if temporal_boundary_mask[det1] and temporal_boundary_mask[det2]: # both endpts in temporal boundary mask -- they are not necessarily boundary crossing tho!! (no d, d-1 check)
            weight_1_chains.append((det1, det2))
            t_coord1 = coords_dict[det1][2] # get the t coordinate of the first endpoint (det1 is the id of the node/detector; coords_dict gives the actual coordinate of the node)
            t_coord2 = coords_dict[det2][2] # get t coord of second endpoint
            min_t, max_t = min(t_coord1, t_coord2), max(t_coord1, t_coord2) # get the min and max t
            # d = code distance. Buffer region = d length, commit region = d. So, boundary btwn buffer & commit is precisely btwn d & d-1. Buffer & commit region are of 2 diff time steps, which is why it's crossing a temporal boundary here
            if min_t == d - 1 and max_t == d: # if this edge falls between 2 temporal measurement rounds (cross exactly one time step across the window boundary), add to one_step_chains
                one_step_chains.append((det1, det2))

    data_dep_srcs = [] # source detectors on commit side
    for det_id in np.where(temporal_boundary_mask)[0]: # only look at temporal boundary mask detectors
        if coords_dict[det_id][2] < d:
            data_dep_srcs.append(det_id) # sources of data dependency, since these IDs are ones that go from commit --> buffer rgn (commit < d, buffer > d)
    weight_2_chains = [] # cross-boundary pairs at weight = 2
    for det1 in data_dep_srcs: # look at sources of data dependencies
        for det2 in np.where(temporal_boundary_mask)[0]: # look at which endpoints are in the temporal boundary
            # print(det1, det2)
            if coords_dict[det2][2] >= d and get_dep_weight(det1, det2) == 2: # if the 2nd endpoint is in the buffer rgn (crosses boundary) and the weight of the edge is 2, then append to edge_2_weight
                weight_2_chains.append((det1, det2))
            if get_dep_weight(det1, det2) is None:
                # print(det1, det2)
                print("IS NONE") # I think these are cases where we try to connect w window before for patch (1,0), but patch (1,0) only spawns here

    match_times = [] # per-shot decode latency
    corrects = [] # boolean per shot --> speculation correct?
    matching.decode_to_matched_dets_array(sampler.sample(1)) # "warms up" PyMatching so that it builds/caches internal C++ structs so that we don't include that initialization cost in timing of the predictors

    samples = sampler.sample(num_shots) # sample # shots -- measure all syndrome bits # shots times

    # get all speculated matchings
    speculated_matchings = speculator(samples.astype(int), weight_1_chains, weight_2_chains, ignore_steps, one_step_chains)
    failure_idx = []
    # verify all of my speculated matchings
    for i in range(num_shots):
        match_time = datetime.datetime.now()
        # matched_dets is bool array, where true = detector k matched to boundary in solution. false = detector k matched to another detector
        # if matched to boundary, that means the next window is dependent on you, so you are a dependent bit and you have to pass forward your dependency information
        # or, if matched to one on commit side and one on buffer side, this also creates a dependency that u need to pass forward
        # if both dependencies on boundary, will cancel out
        matched_dets = matching.decode_to_matched_dets_array(samples[i]) # run actual decoder to get the ground truth on a specific shot
        match_time = (datetime.datetime.now() - match_time).total_seconds() * 1e6 # us
        # verify that the speculation was correct -- TODO annotate
        correct, spec_syndrome, real_syndrome = verify_speculation(samples[i], speculated_matchings[i], matched_dets, coords_dict, 2, d, nx_G)
        if not correct:
            failure_idx.append((samples[i], spec_syndrome, real_syndrome, matched_dets)) # append the shot whose speculation failed, and the speculated and real syndromes, and the ground truth decoder result
        match_times.append(match_time) # add time it took for actual decoder to run (not speculative one)
        corrects.append(correct) # add whether this speculation was correct or not
    # return match_times latencies, corrects array (speculation correct or not), tuple with the ground truth matching as NetworkX object, coords_dict which maps node IDs to actual coordinates on the graph, and failure_idx which shows which shots failed and why/what the exact difference is
    print(corrects.count(True)/len(corrects))
    print(corrects.count(True))
    print(len(corrects))
    return match_times, corrects, (matching.to_networkx(), coords_dict, failure_idx)  

def process_failures(failure_info, d):
    coords_dict = failure_info[1]
    boundary_dets = [det for det, coords in coords_dict.items() if coords[2] == d]
    false_neg = 0
    false_pos = 0
    both = 0
    for i, (sample, spec_syndrome, real_syndrome, real_matches) in enumerate(failure_info[2]):
        original_sample = [sample[det] for det in boundary_dets]
        is_false_neg = False
        is_false_pos = False
        for j, val in enumerate(original_sample):
            if val and not real_syndrome[j] and spec_syndrome[j]:
                is_false_neg = True
            if val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_pos = True
            if not val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_neg = True
            if not val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_pos = True
        if is_false_pos and is_false_neg:
            both += 1
        elif is_false_neg:
            false_neg += 1
        elif is_false_pos:
            false_pos += 1
    return false_neg, false_pos, both


# nx_G: graph of nodes (windows) and edges (0 -> 1 -> 2... -> num_nodes-1)
# adj_pairs: pairs of adjacent edges (right next to each other)
# decode_time: decoding time/latency to decode a window
# spec_time: extra delay added when a misprediction forces you to restart work/rollback
# accuracy: probability that we speculated correctly (speculation accuracy)
# cond_mult: conditional multiplier/fraction of how much prediction accuracy decreases for boundaries adjacent to a misprediction
# strategy: misprediction strategy we're using (optimistic, pessimistic, adjacent)
def strategy_sim(nx_G: nx.DiGraph, adj_pairs: tuple[tuple[int,int]], decode_time: int, spec_time: int, accuracy: float, cond_mult: float, strategy: str) -> tuple[int, int]:
    decoding_queue = {} # dict[int, list[int]] -> map release time, meaning nodes become available at this time
    start_times = nx.get_node_attributes(nx_G, 't') # get the "starting times" for every node/the timestamps
    for node in nx_G.nodes:
        if start_times[node] not in decoding_queue:
            decoding_queue[start_times[node]] = [] # key = timestamp of a node, value = list of nodes with this start time (see below statement)
        decoding_queue[start_times[node]].append(node)
    pred_accuracy = {boundary: accuracy for boundary in nx_G.edges} # get the starting accuracy for each edge. per-edge prediction accuracy. # per-edge, current probability that the speculation on the boundary is correct
    active_rounds = {} # dict[int, list[int]] maps commit time -> list of nodes finishing then
    active_nodes = {} # dict[int, int]: maps node ID -> scheduled commit time (when its current decode run will finish)
    complete_nodes = {} # dict[int, int]: maps node ID -> actual commit time if already finished (for if misprediction leads to un-committing/restarting)
    decode_rounds = 0 # total classical work (sum number of all active nodes during entire decoding process (add num of all active nodes to a rolling sum (decode_rounds) every timestamp/while loop iteration))
    valid_rounds = 0 # accumulate baseline useful work (add decode times of committed nodes (if a committed node is rolled back, will subtract its decode time from this))
    max_proc = 0 # tracks peak concurrency (max number of active decoders/nodes/windows at any tick)
    round = 0 # current time tick (simulation clock)
    while len(decoding_queue) > 0 or len(active_nodes) > 0: # while we still have nodes to decode at some timestamp (future work), or there's currently work in flight (len(acvie_nodes)>0)
        decode_rounds += len(active_nodes) # add number of active nodes to decode_rounds. over time, this sums to total processor-ticks (all classical work done by all nodes at all timeticks)
        max_proc = max(max_proc, len(active_nodes)) # update peak concurrency -- check if current num of active nodes is more than the current peak concurrency number of nodes
        # ensure that active_rounds has a key-value slot for the current round+decode_time (used for poisoned node itself rollback)
        # for nodes that finish normally (so just regular decode_time latency)
        if round + decode_time not in active_rounds:
            active_rounds[round + decode_time] = []
        # ensure that active_rounds has a key-value slot for the current round+spec_time+decode_time (used for nodes that need to be rolled back due to poisoned node)
        # used for nodes that are mispredicted and misprediction triggers a restart/rollback
        if round + decode_time + spec_time not in active_rounds:
            active_rounds[round + decode_time + spec_time] = []
        # commit any nodes scheduled to finish now (they finish at the current round/clock time)
        to_commit = active_rounds[round].copy() if round in active_rounds else []
        for node in to_commit:
            if node not in active_rounds[round]: # need to check, since some nodes might have previously been in active_rounds, but then are removed from the current key=round because of a misprediction
                continue
            # commit the node, by removing from active_rounds and active_nodes
            # add to complete_nodes with the round/clock time it finished as value. Add the decode_time to valid_rounds (baseline useful work -- minimum useful work (so this case is achieved if every node strictly just takes decode_time to finish))
            active_rounds[round].remove(node)
            active_nodes.pop(node)
            complete_nodes[node] = round
            valid_rounds += decode_time
            source_boundries = nx_G.out_edges(node) # iterate through all edges stemming from this node (so all connections to other nodes), since this node is now decoded and we thus need to check if this node was decoded incorrectly due to a misspeculation (and thus rollback) or not
            for (_, dependent) in source_boundries: # dependent is the destination node of the edge
                boundary = (node, dependent) # create the edge tuple with the 2 endpoint nodes
                pred = np.random.choice([True, False], p=[pred_accuracy[boundary], 1 - pred_accuracy[boundary]]) # draw a bernoulli outcome with success probability=the prediction accuracy of the current boundary/edge
                if not pred: # if we mispredicted
                    # handle the directly dependent node
                    if dependent in active_nodes: # dependent currently decoding -> cancel its existing finish and rescheudle to finish at new time
                        active_rounds[active_nodes[dependent]].remove(dependent)
                        active_nodes.pop(dependent)
                        active_rounds[round + decode_time].append(dependent) # only round+decode_time because we'll restart this node, and directly decode it fully using the full decoder because this is the direct dependent (so we'd decode this next anyways with the full decoder)
                        active_nodes[dependent] = round + decode_time 
                    elif dependent in complete_nodes: # dependent already finished -> need to uncommit (complete_nodes+valid_rounds change), and reinsert into active_rounds with new time
                        complete_nodes.pop(dependent)
                        valid_rounds -= decode_time
                        active_rounds[round + decode_time].append(dependent)
                        active_nodes[dependent] = round + decode_time
                    # based on different misprediction strategies, figure out how many nodes to rollback
                    if strategy == 'pessimistic': # restart all descendants
                        descendants = nx.descendants(nx_G, dependent)
                        for restart_node in descendants:
                            if restart_node in active_nodes:
                                active_rounds[active_nodes[restart_node]].remove(restart_node)
                                active_nodes.pop(restart_node)
                                # if active, cancel and reschedule with restart overhead spec_time before decode_time (need spec_time because we need to restart the speculation (we're still speculating for these descendants, we're not yet fully decoding them yet with full decoder))
                                active_nodes[restart_node] = round + spec_time + decode_time # spec_time = extra speculation time that u now wasted bc ur rolling back. decode_time = actual time it takes to fully decode this window properly with full decoder (no speculation). round = current simulation clock time (bc want to store absolute clock time)
                                active_rounds[round + decode_time + spec_time].append(restart_node)
                            elif restart_node in complete_nodes:
                                complete_nodes.pop(restart_node)
                                valid_rounds -= decode_time
                                active_nodes[restart_node] = round + spec_time + decode_time
                                active_rounds[round + decode_time + spec_time].append(restart_node)
                    elif strategy == 'optimistic': # don't restart anyone else; just lower boundary prediction accuracy on adjacent edges/boundaries (found from adj_pairs that we passed in as arg)
                        adj_edges = nx_G.out_edges(dependent)
                        for adj_edge in adj_edges:
                            if (boundary, adj_edge) in adj_pairs:
                                pred_accuracy[adj_edge] = accuracy * cond_mult
                    elif strategy == 'adjacent': # restart only immediate next adjacent nodes (so restart the nodes that are adjacent to the dependent node (dependent node is adjacent to the poisoned node))
                        adj_edges = nx_G.out_edges(dependent) # note dependent here!
                        for adj_edge in adj_edges:
                            if (boundary, adj_edge) in adj_pairs: # TODO figure out why we need to iterate through/check this again I don't get why we ned to check this
                                restart_node = adj_edge[1] # adj_edge[0] is the "source" node, adj_edge[1] is the "receiver/adjacent" node
                                if restart_node in active_nodes:
                                    active_rounds[active_nodes[restart_node]].remove(restart_node)
                                    active_nodes.pop(restart_node)
                                    active_nodes[restart_node] = round + spec_time + decode_time
                                    active_rounds[round + decode_time + spec_time].append(restart_node)
                                elif restart_node in complete_nodes:
                                    complete_nodes.pop(restart_node)
                                    valid_rounds -= decode_time
                                    active_nodes[restart_node] = round + spec_time + decode_time
                                    active_rounds[round + decode_time + spec_time].append(restart_node)
                pred_accuracy[boundary] = 1 # after process boundary, mark as resolved by make prediction accuracy 1 (bc alr fully decoded, so we know it's correct/true)

        # start any new nodes that arrived at this tick (which are using decoding_queue to track arrivals of nodes at some particular tick)
        to_schedule = decoding_queue[round] if round in decoding_queue else []
        for node in to_schedule:
            decoding_queue[round].remove(node) # remove from decoding queue, because now we move it to active nodes/rounds
            if len(decoding_queue[round]) == 0: # if finished with all nodes in this round, remove this round from the decoding queue (cleanup)
                decoding_queue.pop(round)
            active_rounds[round + decode_time].append(node) # add this node to the active data structures. scheduled commit time = round+decode_time because that's when we expect the node to be fully decoded+verified (speculation just changes when nodes start, so basically they can start at an earlier round)
            active_nodes[node] = round + decode_time
            
        round += 1 # increment round bc its our clock
    return round, decode_rounds, valid_rounds, max_proc # return the current clock time, the total amount of classical work/#nodes decoded at every time, baseline useful classical work, maximum parallelization/# parallel ndes running concurrently at the same time