from dataclasses import dataclass, asdict
import networkx as nx
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import copy
import random

@dataclass
class SyndromeRound: # a syndrome round is a complete cycle of measuring all the stabilizer (syndrome) qubits across the surface code lattice
    """A syndrome round for a given patch"""
    patch: tuple[int, int] # spatial coords of the patch (logical qubit rgn) this syndrome measurement corresponds to
    round: int # the measurement round index where this syndrome round corresponds with
    instruction: Instruction # the lattice surgery operation responsible for generating this round (every patch is engaged in one instruction, so this field records which instruction corresponds with this patch in this specific temporal round)
    instruction_idx: int # the index for the instruction that this patch/round corresponds to
    initialized_patch: bool # whether this patch is just initialized/being used for the first time (if patch wasn't acted on before by other inst, this bool is true)
    is_unwanted_idle: bool = False # patch that is unintentionally idle (not being worked on/assigned to an explicit operation)
    discard_after: bool = False # marks patches that are discarded immediately after this round (if a DISCARD frees this patch at the end of this round, then this is true)

    def __repr__(self):
        return f'SyndromeRound({self.patch}, r={self.round}, instr={self.instruction_idx}, init={self.initialized_patch}, discard={self.discard_after})'

@dataclass
class DeviceData: 
    """Data containing the history of a device."""
    d: int # code distance
    num_rounds: int # number of syndrome rounds in this device 
    completed_instructions: int # number of completed lattice surgery instructions 
    total_volume: int # total spacetime volume (units: TODO)
    instructions: list[Instruction] # list of all instrucitons in the scheudle
    instruction_start_times: list[int] # list of all the start times of every instruction (indexes corresopnd with instructions array)
    all_patch_coords: list[tuple[int, int]] # set of all patches that were active during the run (store the coords of each patch)
    syndrome_count_by_round: list[int] # for each round index, the number of SyndromeRound objects (syndrome measurements) generated that round (can have diff numbers bc there's one SyndromeRound per patch)
    instruction_count_by_round: list[int] # for each round index, the number of active instructions in that round
    generated_syndrome_data: list[list[SyndromeRound]] # all (list of) SyndromeRound objects for each round r
    patches_initialized_by_round: dict[int, list[tuple[int, int]]] # number of patches that are initialized in a certain round
    conditioned_decode_wait_times: dict[int, int] # for each instruction index that had conditional decoding dependencies, how many rounds did it need to wait before starting to decode?
    avg_conditioned_decode_wait_time: float # avg wait time (in rounds) for all conditional insts that needed to wait for its dependencies before it could begin decoding
    num_t_gates: int # num t gates

    def to_dict(self): # keys=field names, vals = field values
        return asdict(self)
    
@dataclass
class InstructionTask: # for a certain instruction, store its index, the Instruction obj itself, the start_round, and the end_round of the instruction (instructions can span multiple rounds (based on their duration))
    instruction_idx: int
    instruction: Instruction
    start_round: int
    end_round: int

class OrderedSet:
    def __init__(self, rng = np.random.default_rng()):
        self.rng = rng
        self._data = []
        self._set = set()
    
    def add(self, item, push_front=False): # update data and the set
        if item in self._set:
            if push_front: # put data in the front of the list (which is actually at the largest index of the list here technically)
                self._data.remove(item)
                self._data.append(item)
        else:
            self._data.append(item)
            self._set.add(item)
    
    def pop(self):
        item = self._data.pop() # returns the last element in the list (the one at the largest index of the list) (so this is LIFO)
        self._set.remove(item)
        return item
    
    def pop_random(self): # pop a random thing off of the list
        idx = self.rng.choice(len(self._data))
        item = self._data[idx] # get item that we want to pop off
        self._data[idx] = self._data[-1] # replace the value at the index that we want to remove with the value at the very end of the list
        self._data.pop() # then pop the value at the end of the list, since we've already copied that value and removed the value from the index that was chosen
        self._set.remove(item)
        return item
    
    def __contains__(self, item):
        return item in self._set
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data) # get iterator for the data list that we have here

class DeviceManager:
    def __init__(self, d_t: int, schedule: LatticeSurgerySchedule, lightweight_setting: int = 0, rng: int | np.random.Generator = np.random.default_rng()):
        """TODO

        Args:
            d_t: Temporal distance of the code.
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
        """
        self.d_t = d_t # temporal distance of the code
        self.schedule = schedule.full_schedule() # full schedule of lattice surgery operations to be performed
        self.schedule_instructions = [InstructionTask(i, instr, -1, -1) for i,instr in enumerate(self.schedule.instructions)] # turn each instruction in schedule to an InstructionTask object (retain order)
        self.schedule_dag = self.schedule.to_dag(self.d_t) # turn the schedule into a dag, with distance of d_t (inst durations converted to actual integer values)
        self._patches_initialized_by_instr = self._get_initialized_patches() # returns set of patches initialized by each instruction (every idx in the list corresponds to an inst idx)
        self._is_startup_instruction = [self._calc_is_startup_instruction(i) for i in range(len(self.schedule_instructions))] # for each instruction, is it a startup instruction or not?
        self.current_round = 0 # current round that we're on
        self.lightweight_setting = lightweight_setting # ligthweight settings (for knowing how much metadata info to store)

        self._syndrome_count_by_round = [] # per round, the amount of syndrome measurements we do (or active SyndromeRounds recorded)
        self._instruction_count_by_round = [] # per round, the number of instructions that are active
        self._total_volume = 0 # total spacetime volume
        self._all_patch_coords = set() # coordinates of all patches
        self._generated_syndrome_data = [] # for each round, the syndrome measurement data (SyndromeRounds I think???)
        self._conditional_S_locations = [] # spatial locations (patch coords) where conditional S-gate operations occur
        self._conditioned_decode_wait_times = dict() # for each conditional instruction, the amount of time it had to wait before it could start decoding due to waiting for dependencies
        self._conditioned_decode_wait_time_sum = 0 # total sum of all the wait times in the dict above
        self._conditioned_decode_count = 0 # amount of conditioned instructions there are 
        self._completed_instructions = [] # list of instruction indices that have fully completed execution
        self._completed_instruction_count = 0 # count of completed instructions
        self._active_instructions = dict() # maps instruciton idx --> how many rounds remains before it completes # currently active lattice surgery instructions
        self._active_patches = set() # currently active patches (patches that are currently being used/actively being operated on)

        self._num_T_gates = int # get total number of t gates

        if isinstance(rng, int): # set the rng if needed
            rng = np.random.default_rng(rng)
        self.rng = rng

        # get the true durations based on the current code distance for every single instruction in the schedule of instructions
        self._instruction_durations: list[int] = [Duration.get_true_duration(instr.instruction.duration, self.d_t) for instr in self.schedule_instructions]
        # if we get to a Y_MEAS instruction, and we're conditioned on another instruction; for 50% of the time, we don't need to fixup the instruciton, so we can simply set the fixup instruction duration to 0 (since no fixup is needed)
        # len(instr.instruction.conditioned_on_idx) > 0 means Y_MEAS effect is conditional, meaning it's part of a T-gate injection flow
        for i,instr in enumerate(self.schedule_instructions):
            if instr.instruction.name == 'Y_MEAS' and len(instr.instruction.conditioned_on_idx) > 0 and self.rng.random() < 0: # self.rng.random() < 0.5
                # Conditional S fixup of T injection; 50% chance of not being needed
                # this is all the insts required for an S-correction, but we don't need this S correction 50% of the time because we'll have gotten the correct result
                for idx in instr.instruction.group_instr_indices | {instr.instruction_idx}: # insts grouped with it are the fixup instructions, so we set their duration to 0 since we don't need to do this conditional fixup anymore
                    self._instruction_durations[idx] = 0
                    # self.schedule.instructions[idx].actual_duration_time = 0

        # print("schedule insts device manager after y_meas", self.schedule_instructions)

        # Begin by starting the first instruction
        first_instruction_idx = self._find_first_instruction_idx() # get the index of the first instruction
        self._active_instructions[first_instruction_idx] = self._instruction_durations[first_instruction_idx] # add to the active instructions dictionary, key=first_inst_idx and val=duration of the first_inst_idx
        # orders nodes in topological order (a node that another node depends on will be orderd before the other node), so first generation is nodes with 0 predecessors (which are immediately runnable). 
        # Then, immediate children of the first inst idx can could become runnable after first inst completees. Subtract the first_inst_idx, since this is already being run right now/started/is already chosen
        self._instruction_frontier = (set(next(nx.topological_generations(self.schedule_dag))) | set(self.schedule_dag.successors(first_instruction_idx))) - set([first_instruction_idx]) 
        self._update_active_instructions(set()) # update active insts set (add new insts to active instructions set that are ready to start, complete insts that can be complted (duration=0), start instructions that can be started (must have no dependencies that are still being decoded))

    # return patches that are initialized (first started) by a certain instuction index
    def _get_initialized_patches(self) -> list[set[tuple[int, int]]]:
        """Return the set of patches initialized by each instruction."""
        patches_initialized_by_instr = []
        active_patches = set()
        for i,instr in enumerate(self.schedule_instructions):
            patches = set(instr.instruction.patches)
            # if instruction name is discard, then we want to discard all of these patches from active_patches
            if instr.instruction.name == 'DISCARD':
                # print("discard 1 ", active_patches, patches)
                active_patches -= patches
                patches_initialized_by_instr.append(set()) # for this inst, there's no patches initialiezd by this inst, because it's a DISCARD inst
            # if instruction is not DISCARD, then we append the patches that are initialized by this inst (patches that are not currently active, but become active from/by this inst)
            else:
                patches_initialized_by_instr.append(patches - active_patches)
                active_patches |= patches # add these patches to active_patches now
        return patches_initialized_by_instr

    # an inst is a startup instruction if every single patch that it affects/contains is initialized by it
    def _calc_is_startup_instruction(self, instruction_idx: int) -> bool:
        """Return whether an instruction is a startup instruction."""
        return len(self._patches_initialized_by_instr[instruction_idx]) == len(self.schedule_instructions[instruction_idx].instruction.patches)

    # the first instruction is the first instruction in the longest path in the dag
    def _find_first_instruction_idx(self) -> int:
        schedule_longest_path = nx.dag_longest_path(self.schedule_dag)
        return schedule_longest_path[0]

    # add to first_round dictionary the new instruction_idx and its expected start time
    def _predict_instruction_start_time(self, instruction_idx: int, first_round: dict[int, int]) -> tuple[dict[int, int], list[int]]:
        """Update first_round with the expected start time of
        instruction_idx."""
        # first_round, which is a dict that maps instruction indices → predicted start round
        instructions_to_process = []
        if instruction_idx in first_round:
            pass
        elif instruction_idx in self._active_instructions: # if this inst is an active inst, its start time is its duration+the current round-its duration?? TODO doesn't this just give us the current round?
            start_time = self._active_instructions[instruction_idx] + self.current_round - self._instruction_durations[instruction_idx] # think of it as: (inst_dur-active_inst) gives the number of rnds we've alr decoded, so subtracting curr_rnd-(inst_dur-active_inst) gives the start rnd
            first_round[instruction_idx] = start_time # update first_round with new instruction index and the new start time
        elif self.schedule_instructions[instruction_idx].end_round != -1: # if this instruction has an end round (has ended), then its start time is the end round minus the duration of this idx (+1 bc inclusive of endpts)
            start_time = self.schedule_instructions[instruction_idx].end_round - self._instruction_durations[instruction_idx] + 1
            first_round[instruction_idx] = start_time
        else: 
            valid = True
            expected_start = None
            if self._is_startup_instruction[instruction_idx]:
                # startup instruction; schedule ALAP
                # we want to schedule startup insts as late as possible, since these instructions do not have predecessors in terms of needing earlier work on those patches
                # this inst only needs to happen right before its successors, so we don't want to start it too early because we'd be wasting measurements/might cause conflict with other ops using same patch
                for inst_idx in self.schedule_dag.successors(instruction_idx): # look at successors of this current instruction (ones that are dependent on it)
                    if inst_idx in first_round: # if the successor's start time is already known and is in first_round, which is a dict that maps instruction indices → predicted start round
                        if valid: # valid = haven't encounterd any missing info yet
                            start = first_round[inst_idx] # get the known start time of this instruction
                            this_start = start - self._instruction_durations[instruction_idx] # get latest possible start time of this inst, given that it has to finish before the successor starts (so subtract successor's start minus curr inst's duration to get curr inst's latest start time)
                            if not expected_start or this_start < expected_start: # if the start due to this dependency has to be earlier than the initially predicted expected start, update the expected start to this new start (want to get the earliest start, which is based on the dependencies)
                                expected_start = this_start
                    else: # found an inst that doesn't have a known start time, so our estimate isn't valid, and we need to process this instruction to figure out all of its successors (can only determine inst's start time once all successors are known)
                        expected_start = None
                        valid = False
                        instructions_to_process.append(inst_idx)
            else:
                # standard operation; schedule ASAP
                for inst_idx in self.schedule_dag.predecessors(instruction_idx): # iterate thru all of this inst's predecessors
                    if inst_idx in first_round: # if we know the expected start time of this instruction
                        if valid:
                            start = first_round[inst_idx] # get this inst's estimated start time
                            this_start = start + self._instruction_durations[inst_idx] # we know curr inst must start AFTER its predecessor finishes decoding
                            if not expected_start or this_start > expected_start: # if our current expected start is less than this_start, then update it to be this_start
                                expected_start = this_start
                    elif self._is_startup_instruction[inst_idx]: # if this predecessor instruction is a startup instruction
                        if valid:
                            if self.schedule_instructions[inst_idx].end_round != -1: # if this predecessor inst has an end round, then we should start right after its end round (update expected_start if expected_start is less than this end round, bc then we can't start yet bc this predecessor inst is still decoding)
                                start = self.schedule_instructions[inst_idx].end_round
                                this_start = start
                                if not expected_start or this_start > expected_start:
                                    expected_start = this_start
                            elif inst_idx in self._active_instructions: # or if this predecessor inst is an active instruciton, then we shld start after it finishes (active_insts contains how many rnds are left before this inst finishes decoding, add it to current_rnd to get the rnd when the inst finishes decoding)
                                this_start = self._active_instructions[inst_idx] + self.current_round
                                if not expected_start or this_start > expected_start:
                                    expected_start = this_start
                            else:
                                expected_start = expected_start if expected_start else 0
                    else: # encountered an instruction who we don't know the start of
                        expected_start = None
                        valid = False
                        instructions_to_process.append(inst_idx) # we need to process this predecessor instruction to figure out its start time, which is needed to compute our current inst's start time
            if expected_start is not None:
                expected_start = max(expected_start, self.current_round) # expected start time for our current instruction
                first_round[instruction_idx] = expected_start # update first_round with this inst and its new expected start time
            else:
                instructions_to_process.append(instruction_idx) # append the current instruction index to instructions_to_process, since we know we still have to go back and figure out its expected start time later

        return first_round, instructions_to_process

    def _predict_instruction_start_time_fully(self, instruction_idx: int, first_round: dict[int, int]) -> dict[int, int]:
        """Update first_round with the expected start time of
        instruction_idx, and all instructions it depends on."""
        instructions_to_process = OrderedSet(self.rng) # set with deterministic order and no duplicates (like a stack)
        instructions_to_process.add(instruction_idx) # add curr inst to it
        while len(instructions_to_process) > 0:
            instr = instructions_to_process.pop()
            first_round, new_instructions_to_process = self._predict_instruction_start_time(instr, first_round) # try to get this inst's start time. in the helper function, if we can't get this inst's start time, we'll update the new_instructions_to_process list with a new list of instructions that we have to process/figure out the start time of
            # reverse ensures that dependencies are preserved (operates in correct logical order)
            for new_instr in reversed(new_instructions_to_process): # add these new instructions to process to our current instructions to process list (if we add, we want to make sure that this instruction is pushed to the front of the list (LIFO))
                instructions_to_process.add(new_instr, push_front=True)
        return first_round # return new predictions of estimated start time for this instruction index (and potentially more updates)

    def _predict_instruction_start_times(self):
        """For each not-yet-started instruction in the frontier, get number of
        rounds from now at which we expect it to begin, assuming no unexpected
        delays happen. Instructions which should be started immediately are
        assigned round 0.

        TODO: can re-use most of the results from the previous time this was
        evaluated. Need to check which things have changed since then (iterate
        through each instruction in frontier, check if any of its predecessors
        or successors have changed status since last time, etc.)
        """
        first_round = dict()

        for instruction_idx in self._instruction_frontier: # for every instruction in instruction_frontier, predict its start time
            first_round = self._predict_instruction_start_time_fully(instruction_idx, first_round)
        
        return first_round

    def _update_active_instructions(self, not_fully_decoded_instructions: set[int]) -> None: # instruction_frontier contains insts who are ready/almost ready to run (their dependencies are satisfied)
        """Add new instructions to the active set if they are ready to start.
        Immediately complete any instructions with duration 0. Instructions with
        conditional dependencies cannot be started if any of the instructions
        they are conditioned on are still being decoded.

        Args:
            not_fully_decoded_instructions: Set of instruction indices whose
                data has not yet been fully decoded, whether or not the
                instruction has finished being applied on the device.
        """ 
        # print("instruction frontier update active insts", self._instruction_frontier)
        patches_in_use = set()
        for instruction_idx in self._active_instructions.keys(): # for all active instructions, get all the patches they use (all stored in a set)
            patches_in_use.update(self.schedule_instructions[instruction_idx].instruction.patches)

        waiting_conditional_decode_instructions = set() # instructions that are waiting to start decoding because an inst they're dependent on has not finished decoding
        start_times = self._predict_instruction_start_times() # get all predicted start times of instructions
        done_with_zero_duration_instructions = False
        while not done_with_zero_duration_instructions:
            done_with_zero_duration_instructions = True
            for instruction_idx in self._instruction_frontier: # for all instructions in instruction_forntier
                assert self.schedule_instructions[instruction_idx].start_round == -1 # assert this instruction has not started yet
                if start_times[instruction_idx] <= self.current_round: # if predicted start time for this inst is less than current round
                    instruction_task = self.schedule_instructions[instruction_idx] # we want to try to start this instruction
                    if any(patch in patches_in_use for patch in instruction_task.instruction.patches):
                        # at least one patch is already in use
                        pass
                    elif any(conditioned in not_fully_decoded_instructions for conditioned in instruction_task.instruction.conditioned_on_idx): # any of the insts that it's dependent on are not done decoding
                        # decoding dependency not yet satisfied
                        # instruction needs the decoded measurement outcome(s) of some earlier instruction(s) before it can choose what to do
                        # yes decoding aspect
                        waiting_conditional_decode_instructions.add(instruction_idx) # this inst is now waiting for an inst that it's dependent on
                    elif any(self.schedule_instructions[conditioned].end_round == -1 for conditioned in instruction_task.instruction.conditioned_on_completion_idx): # dependent on other inst being completed, but this inst has not been completed
                        # dependency not yet satisfied
                        # instruction cannot start until some other instruction(s) have fully completed on the device
                        # no decoding aspect
                        pass
                    elif any(self.schedule_instructions[pred].end_round == -1 for pred in self.schedule_dag.predecessors(instruction_idx)): # if any of the predecessors have not been completed
                        # not all predecessors have been completed
                        pass
                    elif self._instruction_durations[instruction_idx] == 0: # if this inst's duration is 0
                        done_with_zero_duration_instructions = False
                        # set this inst's start and end round to the same round (curr_round-1)
                        self.schedule_instructions[instruction_idx].start_round = self.current_round-1
                        self.schedule_instructions[instruction_idx].end_round = self.current_round-1
                        self._completed_instruction_count += 1 # this inst is completed
                        if self.lightweight_setting < 2:
                            self._completed_instructions.append(instruction_idx) # want to store completed insts
                        if instruction_task.instruction.name == 'DISCARD': # note that at any point, only one instructio may operate on a given patch
                            # print("discard 2 ", self._active_patches, set(instruction_task.instruction.patches), instruction_task.instruction, instruction_task.instruction_idx)
                            self._active_patches -= set(instruction_task.instruction.patches) # if is DISCARD inst, need to remove this inst's patches from active patches
                        # log IDLE inside the group
                        elif instruction_task.instruction.name == 'MERGE' and instruction_task.instruction.group_name == 'CONDITIONAL_S' and self.lightweight_setting == 0: 
                            assert len(instruction_task.instruction.conditioned_on_idx) > 0 # assert this instruction is indeed part of a T-gate teleportation
                            idle_instr = self.schedule_instructions[instruction_task.instruction_idx+2] # idle_inst is part of the CONDITIONAL_S group (see lattice surgery scheduler)
                            assert idle_instr.instruction.name == 'IDLE'
                            assert idle_instr.instruction_idx in instruction_task.instruction.group_instr_indices
                            assert idle_instr.instruction.conditioned_on_idx == instruction_task.instruction.conditioned_on_idx # both are in the same T-gate teleportation circuit/group of insts
                            self._conditional_S_locations.append((list(idle_instr.instruction.patches)[0], self.current_round-1)) # append this patch and round number to the conditional_S locations (patch[0] only have one patch anyways, so is fine)
                        # log the Y_MEAS in the group
                        elif instruction_task.instruction.name == 'Y_MEAS' and len(instruction_task.instruction.conditioned_on_idx) > 0 and len(instruction_task.instruction.group_instr_indices) == 0: # is a Y measure -- def part of a conditional S
                            self._conditional_S_locations.append((list(instruction_task.instruction.patches)[0], self.current_round-1)) # add to conditional S locations
                        self._instruction_frontier -= set([instruction_idx]) # subtract this inst from instruction_frontier, bc this inst is now run
                        new_instructions = set(self.schedule_dag.successors(instruction_idx)) - self._instruction_frontier # new instructions are all successors minus any insts on the frontier
                        for instr in new_instructions:
                            start_times = self._predict_instruction_start_time_fully(instr, start_times) # now these insts are launched, so we predict their start times
                        self._instruction_frontier.update(new_instructions) # update instruction frontier with these new instructions
                        break

        for instruction_idx in self._instruction_frontier.copy():
            assert self.schedule_instructions[instruction_idx].start_round == -1
            if start_times[instruction_idx] <= self.current_round:
                instruction_task = self.schedule_instructions[instruction_idx]
                if any(patch in patches_in_use for patch in instruction_task.instruction.patches):
                    # at least one patch is already in use
                    pass
                elif instruction_task.instruction.conditioned_on_idx & not_fully_decoded_instructions:
                    # decoding dependency not yet satisfied
                    waiting_conditional_decode_instructions.add(instruction_idx)
                    pass
                elif any(self.schedule_instructions[conditioned].end_round == -1 for conditioned in instruction_task.instruction.conditioned_on_completion_idx):
                    # dependency not yet satisfied
                    pass
                elif any(self.schedule_instructions[pred].end_round == -1 for pred in self.schedule_dag.predecessors(instruction_idx)):
                    # not all predecessors have been completed
                    pass
                else:
                    assert self._instruction_durations[instruction_idx] > 0
                    self.schedule_instructions[instruction_idx].start_round = self.current_round
                    self._active_instructions[instruction_idx] = self._instruction_durations[instruction_idx]
                    patches_in_use.update(instruction_task.instruction.patches)
                    if instruction_task.instruction.name == 'MERGE' and instruction_task.instruction.group_name == 'CONDITIONAL_S' and self.lightweight_setting == 0:
                        assert len(instruction_task.instruction.conditioned_on_idx) > 0
                        idle_instr = self.schedule_instructions[instruction_task.instruction_idx+2]
                        assert idle_instr.instruction.name == 'IDLE'
                        assert idle_instr.instruction_idx in instruction_task.instruction.group_instr_indices
                        assert idle_instr.instruction.conditioned_on_idx == instruction_task.instruction.conditioned_on_idx
                        self._conditional_S_locations.append((list(idle_instr.instruction.patches)[0], self.current_round-1))
                    elif instruction_task.instruction.name == 'Y_MEAS' and len(instruction_task.instruction.conditioned_on_idx) > 0 and len(instruction_task.instruction.group_instr_indices) == 0:
                        self._conditional_S_locations.append((list(instruction_task.instruction.patches)[0], self.current_round-1))
                    self._instruction_frontier -= set([instruction_idx]) # remove this inst bc it's started execution
                    new_instructions = set(self.schedule_dag.successors(instruction_idx)) - self._instruction_frontier # identify new instructions that may be ready to start becuase one of their dependencies has just been launched (but ensure no overlap w current insts in frontier (no double-counting))
                    for instr in new_instructions: # predict start times for these newly unblocked insts that weren't in instruction frontier
                        start_times = self._predict_instruction_start_time_fully(instr, start_times)
                    self._instruction_frontier.update(new_instructions) # add these new instructions to the frontier 

        # Keep track of how long conditional instructions have to wait
        for instr_idx in waiting_conditional_decode_instructions: # iterate thru all insts that are waiting due to dependencies
            if self.lightweight_setting < 2:
                self._conditioned_decode_wait_times[instr_idx] = self._conditioned_decode_wait_times.get(instr_idx, 0) + 1 # keep track of/update how long an inst is waiting due to dependencies
            self._conditioned_decode_wait_time_sum += 1 # update total wait time sum too

    def _generate_syndrome_round(self) -> tuple[list[SyndromeRound], set[int]]:
        generated_syndrome_rounds = []

        if self.lightweight_setting == 0:
            self._instruction_count_by_round.append(0) # update this only for lowets lightweight setting -- this array keeps track of how many instructions there are per round (and we're on round 0 rn, with 0 insts)
        patches_used_this_round = set()
        completed_instructions = set()
        # print("generate syndrome round device manager ", self._active_instructions)
        for instruction_idx in self._active_instructions.keys():
            instruction_task = self.schedule_instructions[instruction_idx]
            # active_instructions: dict mapping instruction index → number of rounds remaining until that instruction finishes
            # the parantheses stuff is what Python will raise if the asertion fails (assertion checks trivial programming bookeeping expectations)
            assert self._active_instructions[instruction_idx] > 0, (instruction_idx, self._active_instructions[instruction_idx], self.schedule_instructions[instruction_idx], self.schedule_dag.predecessors(instruction_idx), self.schedule_dag.successors(instruction_idx))
            # create a new syndrome round for all the patches in the current instruction
            generated_syndrome_rounds.extend([
                SyndromeRound(coords, 
                              self.current_round, 
                              instruction_task.instruction, 
                              instruction_idx,
                              initialized_patch=(coords not in self._active_patches)) 
                for coords in instruction_task.instruction.patches
                ])
            patches_used_this_round.update(instruction_task.instruction.patches)
            self._active_patches.update(instruction_task.instruction.patches)
            self._active_instructions[instruction_idx] -= 1 # because now we're 1 round closer to finishing this instruction
            if self.lightweight_setting == 0:
                self._instruction_count_by_round[-1] += 1 # this inst was executed, update this bookeeping array
            if self._active_instructions[instruction_idx] == 0:
                completed_instructions.add(instruction_idx) # if this inst now is 0 rounds until it is complete, we add it to the completed instructions array
        if self.lightweight_setting < 2: # update all patch coordinates for bookeeping with patches used in this round
            self._all_patch_coords.update(patches_used_this_round)
        # print("syndrome rounds inactive patches", self._active_patches - patches_used_this_round, self._active_patches, patches_used_this_round)
        # for all coordinates that are in active patches, but are not used in this specific round by this inst, they are in an "unwanted_idle" state. update this.
        generated_syndrome_rounds.extend([
            SyndromeRound(coords, 
                          self.current_round, 
                          Instruction('UNWANTED_IDLE', -1, frozenset([coords]), 1), 
                          -1,
                          initialized_patch=False, 
                          is_unwanted_idle=True) 
            for coords in self._active_patches - patches_used_this_round
            ])
        
        # update bookeeping -- insts in this include all active patches (even ones in unwanted idles -- basically add unwanted idles to this inst count by round bookeeping)
        # append number of generated syndrome rounds to syndrome_count_by_round, and append actual generated_syndrome_rounds data to array
        if self.lightweight_setting == 0:
            self._instruction_count_by_round[-1] += len(self._active_patches - patches_used_this_round)
            self._syndrome_count_by_round.append(len(generated_syndrome_rounds))
            self._generated_syndrome_data.append(generated_syndrome_rounds)
        self._total_volume += len(generated_syndrome_rounds) # update total volume with all the new syndrome rounds generated

        return generated_syndrome_rounds, completed_instructions
    
    # do bookeeping for completed insts
    def _clean_completed_instructions(self, completed_instructions: set[int] = set()):
        for instruction_idx in completed_instructions:
            self.schedule_instructions[instruction_idx].end_round = self.current_round
            self._completed_instruction_count += 1
            if self.lightweight_setting < 2:
                self._completed_instructions.append(instruction_idx)
            self._active_instructions.pop(instruction_idx) # remove completed instructions from active instructions set

    def get_next_round(self, incomplete_instructions: set[int]) -> list[SyndromeRound]:
        """Return another round of syndrome measurements, starting new
        instructions if possible.

        Args:
            incomplete_instructions: Set of instruction indices whose data has
                not yet been fully decoded.
        
        Returns:
            generated_syndrome_rounds: List of SyndromeRound objects for the
                current round.
            discarded_patches: Set of patches that were discarded after the
                current round.
        """
        if self.is_done():
            return []

        init_active_patches = copy.deepcopy(self._active_patches)
        generated_syndrome_rounds, completed_instructions = self._generate_syndrome_round() # get syndrome rounds generated in current round and completed instructions in this round
        self._clean_completed_instructions(completed_instructions) # handle completed instructions

        if not self.is_done(): # if not done yet, increment to next round
            self.current_round += 1

        self._update_active_instructions(incomplete_instructions) # update active instructions with all incomplete instructions (see if can start any of these incomplete insts, etc)
        discarded_patches = init_active_patches - self._active_patches # discarded patches are initially active patches minus patches that are still active now
        for dp in discarded_patches: # NOTE: there shouldb e ones yndrome round per patch per round
            syndrome_round = [sr for sr in generated_syndrome_rounds if sr.patch == dp][0] # if syndrome round is affected by this discarded patch, then we take this syndrome round bc this is the syndrome round of the discarded patch
            syndrome_round.discard_after = True # this syndrome round should be discarded right after this round (is the last round)
    
        return generated_syndrome_rounds
    
    def is_done(self) -> bool:
        """Return whether all instructions have been completed."""
        # frontier=0 means no more instructions that are ready to start
        # active instructions=no more insts currently active
        # completed insts=all insts in schedule --> all insts done
        return len(self._instruction_frontier) == 0 and len(self._active_instructions) == 0 and self._completed_instruction_count == len(self.schedule_instructions)
    
    def _postprocess_idle_data(self, syndrome_data: list[list[SyndromeRound]]) -> list[list[SyndromeRound]]:
        """Rename UNWANTED_IDLE syndrome rounds to either DECODE_IDLE (if they
        happen before a conditional gate, while waiting for a decode) or IDLE
        (otherwise).
        """
        # collect continuous groups of syndrome data for each patch
        data_by_patch = {patch: [[]] for patch in self._all_patch_coords} # for each patch, collects a list of lists, where each inner list is separated by round (TODO I THINK??? very unsure)
        for round_idx,round_data in enumerate(syndrome_data): # syndrome_data contains all syndromerounds in a specific round
            used_patches = set()
            for i,sr in enumerate(round_data):
                used_patches.add(sr.patch)
                data_by_patch[sr.patch][-1].append((sr, round_idx, i)) # append to syndrome-round entry to the most recent segment for this patch 
            for patch in self._all_patch_coords - used_patches: # for patches that are not used
                if len(data_by_patch[patch]) > 0: # if this patch already has data in it
                    data_by_patch[patch].append([]) # simply append an empty list, since we don't use it in this current round

        # look at each patch's data, which contains the syndrome round, round_idx, and i
        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for j,(sr,_,_) in enumerate(continuous_data):
                    if sr.is_unwanted_idle: # if this syndrome round is an unwanted idle, then we rename the instruction to an IDLE instructio instead
                        patch_data[i][j][0].instruction = sr.instruction.rename('IDLE')

        # check for decode idles (based on conditional S locations)
        # start at S gate meadsurement and walk backward to upgrade idles to DECODE_IDLE
        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for j,(sr,_,_) in enumerate(continuous_data):
                    if (sr.patch, sr.round) in self._conditional_S_locations:
                        for jj in range(j,-1,-1):
                            if patch_data[i][jj][0].is_unwanted_idle: # walk backwards thru all data in a certain patch's list, see if it's an unwanted idle and if the patch/round is a conditional S, if so then we rename the inst to a DECODE_IDLE
                                patch_data[i][jj][0].instruction = patch_data[i][jj][0].instruction.rename('DECODE_IDLE')
                                # data_by_patch[patch][i][jj] = (sr, round_idx, sr_idx, 'DECODE_IDLE')
                            else:
                                break

        # reconstruct syndrome data
        new_syndrome_data = [[None for _ in syndrome_data[i]] for i in range(len(syndrome_data))]
        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for sr,round_idx,sr_idx in continuous_data:
                    assert sr.instruction.name != 'UNWANTED_IDLE' # we don't want anymore unwanted_idles in our new syndrome data (shoould've gotten rid of all of them by now)
                    new_syndrome_data[round_idx][sr_idx] = sr # replace new syndrome data with insts with new types of idles

        return new_syndrome_data

    def get_data(self):
        """Return all relevant data regarding device history."""        
        patches_initialized_by_round = {round_idx: set() for round_idx in range(self.current_round+2)}
        for instr, round_idx in self._predict_instruction_start_times().items():
            if round_idx <= self.current_round:
                patches_initialized_by_round[round_idx] |= self._patches_initialized_by_instr[instr] # add the patches initialized by the instruction to the patches initialized by round bookeeping array

        conditional_S_count = sum(1 for instr in self.schedule_instructions if instr.instruction.name == 'Y_MEAS' and len(instr.instruction.conditioned_on_idx) > 0) # Y_MEAS that's part of a group is a conditional S -- add to bookeeping

        num_t_gates = 0
        for inst in self.schedule_instructions:
            if inst.instruction.name == 'INJECT_T':
                num_t_gates = num_t_gates + 1

        if self.lightweight_setting == 0:
            return DeviceData(
                d=self.d_t, # code dist
                num_rounds=self.current_round, # current round we're on
                completed_instructions=self._completed_instruction_count, # num completed insts
                total_volume=self._total_volume, # total volume
                instructions=[instr.instruction for instr in self.schedule_instructions], # all insts
                instruction_start_times=[(instr_task.end_round-self._instruction_durations[i]+1 if instr_task.end_round != -1 else None) for i,instr_task in enumerate(self.schedule_instructions)], # start times of insts that have ended
                all_patch_coords=list(self._all_patch_coords),
                syndrome_count_by_round=self._syndrome_count_by_round, # number of syndromerounds per round
                instruction_count_by_round=self._instruction_count_by_round, # num instructions per round
                generated_syndrome_data=self._postprocess_idle_data(self._generated_syndrome_data), # all syndrome data
                patches_initialized_by_round={k: list(v) for k,v in patches_initialized_by_round.items()}, # patches initialized in a certain round
                conditioned_decode_wait_times=self._conditioned_decode_wait_times, # amt of time an inst has spent waiting due to its dependencies
                avg_conditioned_decode_wait_time=self._conditioned_decode_wait_time_sum / conditional_S_count if conditional_S_count > 0 else 0, # average decode wait time (average taken based on conditional_S_count)
                num_t_gates=num_t_gates,
            )
        # same as above, just store less stuff
        elif self.lightweight_setting == 1:
            return DeviceData(
                d=self.d_t,
                num_rounds=self.current_round,
                completed_instructions=self._completed_instruction_count,
                total_volume=self._total_volume,
                instructions=None,
                instruction_start_times=[(instr_task.end_round-self._instruction_durations[i]+1 if instr_task.end_round != -1 else None) for i,instr_task in enumerate(self.schedule_instructions)],
                all_patch_coords=list(self._all_patch_coords),
                syndrome_count_by_round=None,
                instruction_count_by_round=None,
                generated_syndrome_data=None,
                patches_initialized_by_round=None,
                conditioned_decode_wait_times=self._conditioned_decode_wait_times,
                avg_conditioned_decode_wait_time=self._conditioned_decode_wait_time_sum / conditional_S_count if conditional_S_count > 0 else 0,
                num_t_gates=num_t_gates,
            )
        # same as above, just store less stuff
        elif self.lightweight_setting == 2 or self.lightweight_setting == 3:
            return DeviceData(
                d=self.d_t,
                num_rounds=self.current_round,
                completed_instructions=self._completed_instruction_count,
                total_volume=self._total_volume,
                instructions=None,
                instruction_start_times=None,
                all_patch_coords=None,
                syndrome_count_by_round=None,
                instruction_count_by_round=None,
                generated_syndrome_data=None,
                patches_initialized_by_round=None,
                conditioned_decode_wait_times=None,
                avg_conditioned_decode_wait_time=self._conditioned_decode_wait_time_sum / conditional_S_count if conditional_S_count > 0 else 0,
                num_t_gates=num_t_gates,
            )
        else:
            raise ValueError('Invalid lightweight setting')
        
    def get_instructions(self):
        return [instr.instruction for instr in self.schedule_instructions]