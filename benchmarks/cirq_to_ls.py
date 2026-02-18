import json
import parse
import re
import subprocess
import os
import shutil
import cirq
import numpy as np
from qualtran.bloqs.mcmt.and_bloq import And

from swiper.lattice_surgery_schedule import LatticeSurgerySchedule

class Cell: # one lattice surgery "patch" (handles geometric/spatial reasoning)

    def __init__(self, slice_idx, row, col, cell_info):
        self.slice_idx = slice_idx # slice idx corr to time
        self.row = row # row, col is spatial
        self.col = col
        self.activity = cell_info['activity']['activity_type'] # activity type e.g. Measurement, Unitary, Idle
        self.patch_type = cell_info['patch_type'] # patch type = qubit, ancilla, DistillationQubit, Empty
        if self.patch_type == 'Qubit': # data qubit, merge check operation diff from ancilla qubit
            self.qubit_id = parse.parse('Id: {}', cell_info['text'])[0]
            is_m = lambda edge: 'Stiched' in edge # rip typo
        elif self.patch_type == 'Ancilla': # ancilla qubit
            is_m = lambda edge: 'Join' in edge # is_m is fcn checking whether it's a part of merge
        else:
            is_m = lambda _: False
        # check if any of the faces are merge
        self.bottom_m = is_m(cell_info['edges']['Bottom'])
        self.left_m = is_m(cell_info['edges']['Left'])
        self.right_m = is_m(cell_info['edges']['Right'])
        self.top_m = is_m(cell_info['edges']['Top'])
    
    def __repr__(self) -> str: # convert to json representation
        return json.dumps(self, 
                          default=lambda o: o.__dict__, 
                          indent=4)

class LLInstruction: # low-level instruction (describe operations performed at each time slice (not spatial))

    def __init__(self, slice_idx, label, args): # only have time and label of inst and args of inst
        self.slice_idx = slice_idx
        self.label = label
        self.args = args
    
    def __repr__(self) -> str:
        return json.dumps(self, 
                          default=lambda o: o.__dict__, 
                          indent=4)
    
# map letter to actual cirq gate
S_GATE_MAPPING = {
    'H' : cirq.H,
    'I' : cirq.I,
    'S' : cirq.S,
    'X' : cirq.X,
    'T' : cirq.T,
    'Z' : cirq.Z,
}

from retry import retry

@retry(exceptions=(subprocess.CalledProcessError), tries=3, delay=2)
def run_subprocess_with_retry(command): # retry by running specific subprocess command (for running gridsynth on unspecified rotation angles)
    try:
        result = subprocess.run(command, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        raise

# given a (single-qubit) op and a target Z-axis rotation angle rads, return a Clifford+T gate sequence that implements the rotation
def _get_gridsynth_sequence(op: cirq.Operation, rads: float, precision: float = 1e-10):
    assert len(op.qubits) == 1 # must be single-qubit gate
    # pi / x = rz_angle => x = pi / rz_angle
    pi_rots = [0.0, np.pi, np.pi / 2, np.pi / 4]
    output_gate = None
    for i, pi_rot in enumerate(pi_rots):
        if abs(abs(rads) - pi_rot) < 1e-10: # check if the rotation angle is equal to one of our 4 set angles, with 1e-10 tolerance/fluctuation
            match i:
                case 0: # 0 = identity -> no gates
                    return []
                # for different rotation angles, returns an exact gate
                case 1:
                    output_gate = cirq.Z
                case 2:
                    output_gate = cirq.S
                case 3:
                    output_gate = cirq.T
            break
    if not output_gate: # if we don't match with any pre-set angle
        # run external gridsynth binary that synthesizes a Clifford+T gate sequence approximating Rz(rads) to error <= precision (default 1e-10)
        angle_str = f'{rads}' if rads >= 0 else f'({rads})'

        command = ["benchmarks/gridsynth", angle_str, f'--epsilon={precision}']
        output = run_subprocess_with_retry(command)

        # first strip the Bytes unit, then remove leading/trailing W symbol, then reverse bc we want it in application order (but it prints in reverse order (last gate first))
        ss = str(output)[2:-3].strip('W')[::-1] 

    
        # Merge double S gate
        # create new sequence with merged double S gate (ss = z)
        new_s = ''
        i = 0
        while i < len(ss):
            if i < len(ss) - 1 and (ss[i] == ss[i+1] == 'S'): # we know s^2 = z
                new_s += 'Z'
                i += 2
            else:
                new_s += ss[i]
                i += 1
        
        # Build a circuit from this
        approx_seq = []
        for s in new_s:
            approx_seq.append(S_GATE_MAPPING[s].on(op.qubits[0])) # translate every gate to a cirq gate object operating on the same qubit (only 1 qubit gate here)
        return approx_seq
    else:
        return [output_gate.on(op.qubits[0])] # if not use gridsynth, we know directly corresponds to a single gate, so just convert this gate to cirq gate object on same qubit

def _get_merge(cell_program, endpoint: Cell): 
    if endpoint.patch_type != 'Qubit':
        raise Exception('Endpoint must be data qubit cell')
    merge_faces = set() # collects all merge edges between connected cells # adjacency pairs btwn merge cells
    # step will walk along the merge face/boundary
    def step(coords: tuple[int,int], from_dir: str):
        cell = cell_program[coords[0]][coords[1]][slice]
        if cell.bottom_m and not from_dir == 'top':
            merge_faces.add(((coords[0], coords[1]), (coords[0]+1, coords[1])))
            return (cell.row+1, cell.col), 'bottom'
        elif cell.top_m and not from_dir == 'bottom':
            merge_faces.add(((coords[0], coords[1]), (coords[0]-1, coords[1])))
            return (cell.row-1, cell.col), 'top'
        elif cell.right_m and not from_dir == 'left':
            merge_faces.add(((coords[0], coords[1]), (coords[0], coords[1]+1)))
            return (cell.row, cell.col+1), 'right'
        elif cell.left_m and not from_dir == 'right':
            merge_faces.add(((coords[0], coords[1]), (coords[0], coords[1]-1)))
            return (cell.row, cell.col-1), 'left'
        return None
    # start at endpoint
    slice = endpoint.slice_idx
    curr = (endpoint.row, endpoint.col)
    data_cells = [curr] # all data qubits conected via merge
    routing_cells = [] # all ancilla qubits connected via merge
    from_dir = None
    while ret := step(curr, from_dir): # step through
        curr, from_dir = ret
        if cell_program[curr[0]][curr[1]][slice].patch_type == 'Ancilla':
            routing_cells.append(curr)
        elif cell_program[curr[0]][curr[1]][slice].patch_type == 'Qubit':
            data_cells.append(curr)
        else:
            raise Exception('Unexpected merged cell type found')
    # check if this merge is a magic state injection
    is_inject = (False, None, None)
    if slice < 2: # not enough history to determine if is injection
        return data_cells, routing_cells, merge_faces, is_inject
    for data in data_cells:
        # check what this cell was doing 2 and 1 time steps/slices back
        old_t_data = cell_program[data[0]][data[1]][slice - 2]
        old_s_data = cell_program[data[0]][data[1]][slice - 1]
        curr_activity = cell_program[data[0]][data[1]][slice].activity
        if curr_activity != 'Measurement':
            continue
        # curr activity must be measurement
        # if Unitary operation before and now measured, likely Y-state injection
        if old_s_data and old_s_data.patch_type == 'Qubit' and old_s_data.activity == 'Unitary':
            # There does not seem to be a better way to check for S gate injection. 
            # Corresponding slices between lli and json seems very unreliable 
            # (e.g. Y state injection is two slices in lli but T state injection is 1 slice in lli
            #  both are 2 slices in json)
            # TODO: See if this creates problematic edge cases
            is_inject = (True, 'Y', data) 
            break
        # if DistillationQubit 2 steps back, then likely is T-state injection
        elif old_t_data and old_t_data.patch_type == 'DistillationQubit':
            is_inject = (True, 'T', data)
            break
        # data_id = cell_program[data[0]][data[1]][slice].qubit_id
        # for instr in lli_program[slice]: # Why is T not injected in the previous timeslice, but Y is
        #     if instr.label == 'RequestMagicState':
        #         for arg in instr.args:
        #             if data_id == arg.split(' ')[0]: #{qubit_id} {_}
        #                 is_inject = (True, 'T', data)
        #                 break
        # for instr in lli_program[slice - 1]:
        #     if instr.label == 'RequestYState':
        #         for arg in instr.args:
        #             if data_id == arg.split(' ')[0]: #{qubit_id} {_}
        #                 is_inject = (True, 'Y', data)
        #                 break
            
    return data_cells, routing_cells, merge_faces, is_inject



def cirq_to_ls(circ: cirq.Circuit, eps=1e-10) -> LatticeSurgerySchedule:
    def decomp(op: cirq.Operation) -> cirq.OP_TREE: # takes op and returns flat list of simpler ops, but simpler ops must be of length <= 2
        return cirq.decompose(op, keep=lambda op: len(op.qubits) <= 2)
    def map_approx_rz(op: cirq.Operation) -> cirq.OP_TREE: # convert all rotations to Rz then Clifford+T sequence of operations
        is_rot = False
        prefix = []
        suffix = []
        if isinstance(op.gate, cirq.ZPowGate):
            is_rot = True
            op = cirq.Rz(rads=op.gate._exponent * np.pi).on(op.qubits[0])
        if isinstance(op.gate, cirq.XPowGate):
            is_rot = True
            op = cirq.Rz(rads=op.gate._exponent * np.pi).on(op.qubits[0])
            prefix = [cirq.H.on(op.qubits[0])]
            suffix = [cirq.H.on(op.qubits[0])]
        if isinstance(op.gate, cirq.YPowGate):
            is_rot = True
            op = cirq.Rz(rads=op.gate._exponent * np.pi).on(op.qubits[0])
            prefix = [cirq.S.on(op.qubits[0]), cirq.H.on(op.qubits[0])]
            suffix = [cirq.H.on(op.qubits[0]), cirq.S.on(op.qubits[0])]
        if isinstance(op.gate, cirq.HPowGate) and op.gate._exponent == -1.0:
            # Manual handling of Gate: H**-1.0
            # ry(pi*0.25)
            # rx(pi*-1.0)
            # ry(pi*-0.25)
            is_rot = True
            # manual set of cirq operations
            return [cirq.S.on(op.qubits[0]), cirq.H.on(op.qubits[0]), cirq.T.on(op.qubits[0]), cirq.H.on(op.qubits[0]), cirq.S.on(op.qubits[0]),
                    cirq.H.on(op.qubits[0]), cirq.Z.on(op.qubits[0]), cirq.H.on(op.qubits[0]),
                    cirq.S.on(op.qubits[0]), cirq.H.on(op.qubits[0]), cirq.T.on(op.qubits[0]), cirq.H.on(op.qubits[0]), cirq.S.on(op.qubits[0])]
        if not is_rot:
            return [op]
        return prefix + [_get_gridsynth_sequence(op, op.gate._rads, precision=eps)] + suffix # get the full rotation angle
    def make_qasm_compat(op: cirq.Operation) -> cirq.OP_TREE: # strips classical controls
        return op.without_classical_controls()

    # convert circuit's ops to only be ones specified by our functions defined above
    circ = circ.map_operations(decomp).map_operations(make_qasm_compat).map_operations(map_approx_rz) 

    # go through every qubit in circuit, and every operation on those qubits, and ensure that the operation is compatible with Qasm -- remove uncompatible ops
    qbit_mapping = {q: f'q_{i}' for i, q in enumerate(circ.all_qubits())}
    bad_ops = []
    for i, moment in enumerate(circ.moments):
        for op in moment:
            no_control_op = op.without_classical_controls()
            try:
                test_qasm = no_control_op._qasm_(cirq.QasmArgs(qubit_id_map=qbit_mapping))
                assert(test_qasm)
            except Exception:
                bad_ops.append((i, op))
    circ.batch_remove(bad_ops)

    # run lattice surgery slicer to copmile to compiled.json, which encodes a 3D spacetime grid (slice x row x col + describe patch, activity, and edge tag)
    os.makedirs('benchmarks/tmp')
    circ.save_qasm('benchmarks/tmp/prog.qasm')
    #subprocess.call(['benchmarks/lsqecc_slicer', '-q', '-i', 'benchmarks/tmp/prog.qasm', '-L', 'edpc', '--disttime', '1', '--nostagger', '-P', 'wave', '--printlli', 'sliced', '-o', 'benchmarks/tmp/lli.txt'])
    subprocess.call(['benchmarks/lsqecc_slicer', '-q', '-i', 'benchmarks/tmp/prog.qasm', '-L', 'edpc', '--disttime', '1', '--nostagger', '-P', 'wave', '-o', 'benchmarks/tmp/compiled.json'])

    prog_data = json.load(open('benchmarks/tmp/compiled.json', 'rb'))
    #prog_instrs = open('benchmarks/tmp/lli.txt', 'r').readlines()

    # lli_program = []
    # for slice_idx, slice_data in enumerate(prog_instrs):
    #     lli_program.append([])
    #     instrs = slice_data.split(';')
    #     for instr in instrs:
    #         try:
    #             label, arg_list = parse.parse('{} {}', instr)
    #             args = re.split(',(?!\d+\))', arg_list) # look-ahead to ignore commas in location tuples
    #             lli_program[slice_idx].append(LLInstruction(slice_idx, label, args))
    #         except:
    #             continue

    # have a cell major and slice major schedule
    # TODO fully understand the mapping!
    cell_major_program = [[[None for _ in range(len(prog_data))]            
                            for _ in range(len(prog_data[0]))]
                            for _ in range(len(prog_data[0][0]))]
    slice_major_program = [[[None for _ in range(len(prog_data[0][0]))]            
                            for _ in range(len(prog_data[0]))]
                            for _ in range(len(prog_data))]

    for i, slice in enumerate(prog_data):
        for r, row in enumerate(slice):
            for c, cell in enumerate(row):
                if cell:
                    cell_major_program[r][c][i] = Cell(i, r, c, cell)
                    slice_major_program[i][r][c] = cell_major_program[r][c][i]
    
    # Ok, data processing done. Convert all instructions 
    schedule = LatticeSurgerySchedule()
    pending_t_inject = {}
    for slice_idx, slice in enumerate(slice_major_program):
        processed_cells = []
        for r, row in enumerate(slice):
            for c, cell in enumerate(row):
                if cell and cell.patch_type == 'Qubit' and (r, c) not in processed_cells:
                    if cell.bottom_m or cell.left_m or cell.right_m or cell.top_m:
                        data, routing, merge_faces, (is_inject, inject_type, inject_cell) = _get_merge(cell_major_program, cell)
                        if is_inject:
                            if len(data) != 2:
                                raise Exception(f'Expected two cells for injection operation, got {len(data)}.')
                            other_cell = [cell for cell in data if cell != inject_cell][0]
                            if inject_type == 'T':
                                schedule.inject_T([inject_cell])
                                merge_idx = schedule.merge(data, routing, merge_faces)
                                if other_cell in pending_t_inject and pending_t_inject[other_cell][0]:
                                    raise Exception("Can't inject T before S gate applied")
                                pending_t_inject[other_cell] = (True, merge_idx)
                                schedule.discard([inject_cell])
                            elif inject_type == 'Y':
                                if other_cell in pending_t_inject and pending_t_inject[other_cell][0]:
                                    # Conditional S after T injection
                                    schedule.S(other_cell, inject_cell, pending_t_inject[other_cell][1])
                                    pending_t_inject[other_cell] = (False, None)
                                else:
                                    # Non-conditional S gate
                                    schedule.S(other_cell, inject_cell)
                        else:
                            schedule.merge(data, routing, merge_faces)
                            for data_coords in data:
                                data_cell = cell_major_program[data_coords[0]][data_coords[1]][slice_idx]
                                if data_cell.activity == 'Measurement':
                                    schedule.discard([data_coords])
                        processed_cells.extend(data)
                        processed_cells.extend(routing)
                    elif cell.activity == 'Measurement':
                        schedule.discard([cell])
                        processed_cells.append(cell)


    # Discard remaining data qubits
    for r, row in enumerate(cell_major_program):
        for c, cell_history in enumerate(row):
            if cell_history[-1] and cell_history[-1].patch_type == 'Qubit' and cell_history[-1].activity != 'Measurement':
                schedule.discard([(r,c)])

    shutil.rmtree('benchmarks/tmp', ignore_errors=True)

    return schedule
                        






