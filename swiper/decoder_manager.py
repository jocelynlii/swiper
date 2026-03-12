from dataclasses import dataclass, field, asdict
from typing import Callable
import numpy as np
import networkx as nx
import itertools
from swiper.window_builder import DecodingWindow
from swiper.lattice_surgery_schedule import Instruction, Duration
from collections import defaultdict, deque

@dataclass
class DecoderData:
    num_rounds: int # number of simulation rounds, where each round is a round of measurement of all ancilla qubits
    max_parallel_decoders: int # max num of parallel decoders active in a single round
    max_parallel_speculators: int # max num of parallel speculators active in a single round
    max_parallel_combined_processes: int # max # of parallel decoders and speculators in a given round (combined)
    decode_process_volume: int # sum of active decoders over all rounds (total decoder-time product)
    speculate_process_volume: int # sum of all speculator processes over all ronuds (total speculator-time product)
    num_completed_windows: int # all decoding windows that have finished decoding by the end of the simulation
    decode_processes_by_round: list[int] # num active decoders per round
    speculate_processes_by_round: list[int] # num speculators per round
    completed_windows_by_round: list[int] # num completed windows per round (cumulative count -- is the # of completed windows per round)
    window_speculation_start_times: dict[int, int] # maps window idx --> round when speculation began for that window
    window_decoding_start_times: dict[int, int] # maps window idx --> round when decoding of that window started
    window_decoding_completion_times: dict[int, int] # maps window idx --> round when decoding of that window finished
    missed_speculation_events: list[tuple[int, list[int]]] # list of (round, list of poisoned window indices) tuples # records each round, and the windows that are poisoned (proven incorrect) in that round
    num_failed_speculations: int # total num of speculation failures/mispredictions that occurred -- each corresopnds to a poisoned speculation event
    num_discarded_decodes: int # total num of decodes that had to be discarded/restarted due to incorrect speculated dependencies --> represents wasted compute
    wasted_decode_volume: int # cycles * decoder # total wasted work in decoder-rounds (sum of elapsed rounds for all decoders that were reset)
    num_successful_speculations: int # total num of speculations that were later verified as correct
    per_window_poisoned: list[int]
    per_window_wasted_rounds: list[int]
    per_window_parent_inst: list[frozenset[int]]
    per_window_spec_acc: dict
    per_inst_windows: dict

    def to_dict(self):
        return asdict(self)

@dataclass
class DecoderTask:
    window: DecodingWindow # actual decoding window object, is a window
    window_idx: int # window index in DAG
    completed_decoding: bool = False # True when this window's decoding is complete
    decoding_start_time: int = -1 # round num when decoding began for this window
    decoding_completion_time: int = -1 # round num when decoding finished for this window
    completed_speculation: bool = False # True once speculation about this window's boundary outcomes has completed (refers to outgoing speculation)
    speculation_start_time: int = -1 # round num when speculation started
    speculation_completion_time: int = -1 # round num when speculation finished
    used_parent_speculations: dict[int, bool] = field(default_factory=dict) # maps parent window idx --> bool, indicating whether this window used that parent's speculation result (True) or the verified decoding result (False)
    speculation_modifiers: dict[int, float] = field(default_factory=dict) # maps child window idx --> modifier factor (float), which increases that child's speculation failure probability due to adjacency with a recently poisoned speculation
    # window_speculation_accuracy: int = 0.9 # ADDED -- speculation accuracy dependent on each window

class DecoderManager:
    def __init__(
            self,
            instruction_idx_dag: nx.DiGraph, # dag of all the instructions (lattice surgery insts)
            decoding_time_function: Callable[[int], int],
            speculation_time: int, # time it takes to speculate (in rounds)
            speculation_accuracy: float,
            max_parallel_processes: int | None = None,
            speculation_mode: str | None = 'integrated', # TODO so if integrated, when can we start speculating? is this saying that the decoder starts "processing" a window when it starts speculating? bc speculation should come before the decoder, right?
            poison_policy: str = 'successors',
            missed_speculation_modifier: float = 1.4,
            lightweight_setting: int = 0, # how much data to return
            rng: int | np.random.Generator = np.random.default_rng(),
            instructions: list[Instruction] = None, # ADDED because want this list in decoder manager too for identifying t-gate instructions
            full_window_dag: nx.DiGraph | None = None, # ADDED -- full window dag used for finding dist to t-gate
        ) -> None:
        """Initialize the decoder manager.
        
        Args:
            decoding_time_function: A function that returns the number of rounds
                required to decode a given spacetime volume of syndrome
                measurements. The volume is specified in units of rounds*d^2.
            speculation_accuracy: Accuracy of the speculation step
            speculation_time: Number of rounds required to make a speculative
                prediction for artifial defects at the boundary of a decoding
                window.
            max_parallel_processes: Maximum number of parallel decoding processes
                to run. If None, run as many as possible and keep track of
                the maximum number that were run.
            speculation_mode: 'integrated', 'separate', or None. If 'integrated', the
                speculation time is included in the decoding time of a window,
                and speculation can only be performed once the decoder starts
                processing the window. If 'separate', the speculation time is
                not included in the decoding time of a window, and speculation
                can be run independently of decoding. In this case, speculation
                uses a parallel process and counts towards
                max_parallel_processes. If None, no speculation is performed.
            poison_policy: 'successors' or 'descendants'. If 'successors', a
                poisoned speculation will reset only direct descendants that
                depended on the speculation. If 'descendants', a poisoned
                speculation will reset all descendants of the poisoned window,
                regardless of whether they directly depended on the speculation.
            missed_speculation_modifier: Factor by which incorrect speculation
                rate increases when an adjacent face has a missed speculation.
            rng: Random number generator, or integer seed.
        """
        # initialize all of our data structs/fields
        self.instruction_idx_dag = instruction_idx_dag
        self.decoding_time_function = decoding_time_function
        # print("init decoding time function", self.decoding_time_function)
        self.speculation_time = speculation_time
        self.speculation_accuracy = speculation_accuracy
        if speculation_mode not in ['integrated', 'separate', None]:
            raise ValueError('Invalid speculation mode')
        self.speculation_mode = speculation_mode
        if poison_policy not in ['successors', 'descendants']:
            raise ValueError('Invalid poison policy')
        self.poison_policy = poison_policy
        self.missed_speculation_modifier = missed_speculation_modifier
        self.lightweight_setting = lightweight_setting
        if isinstance(rng, int): # create rng if given a seed
            rng = np.random.default_rng(rng)
        self.rng = rng

        self.max_parallel_processes = max_parallel_processes
        self._max_speculation_processes_used = 0
        self._max_decoding_processes_used = 0
        self._max_combined_processes_used = 0
        self._decode_processor_spacetime_volume = 0
        self._speculate_processor_spacetime_volume = 0
        self._decode_processes_by_round: list[int] = []
        self._speculate_processes_by_round: list[int] = []
        self._completed_windows_by_round: list[int] = []
        self._num_completed_windows = 0
        self._current_round = 0
        self._missed_speculation_events: list[tuple[int, list[int]]] = []
        self._active_speculation_progress: dict[int, int] = {} # maps each speculated window to rounds remaining to complete speculation
        self._active_window_progress: dict[int, int] = {} # maps each active window to rounds remaining to complete decoding
        self._pending_decode_tasks: set[int] = set() # set of pending decoding tasks that are eligible to start decoding but are not yet running, each entry is the window idx of a DecodingTask
        self._pending_speculate_tasks: set[int] = set() # set of windows waiting to perform speculation (need to predict their boundary outcomes before/during decoding)
        self._tasks_by_idx: list[DecoderTask | None] = [] # index=window idx, at each index is the DecoderTask corresponding to that decoding window's window idx
        self._instruction_tasks: dict[int, set[int]] = {} # maps each instruction to the set of tasks that cover it (tasks=decoding tasks here)
        self._instruction_unverified_task_counts: dict[int, int] = {} # maps each instruction to the number of unverified tasks that depend on it
        self._seen_instructions: set[int] = set() # set of inst indexes that the decoder has encountered so far --> lattice surgery opertations for which at least one decoding window has been created
        self._not_fully_decoded_instructions = set() # insts that are not fully decoded yet/complete (at least one of their decoding windows is not yet verified)
        self._decoded_unverified_tasks: set[int] = set() # windows who have finished decoding but are not yet verified
        self._num_failed_speculations: int = 0 # num of mispredictions
        self._num_discarded_decodes: int = 0 # num of decoding procs that had to be restarted bc of misprediction
        self._wasted_decode_volume: int = 0 # wasted decoding volume (sum of # rounds for each decoder that were wasted due to misprediction)
        self._num_successful_speculations: int = 0

        self._unwanted_idle_rounds: int = 0 # number of unwanted idle rounds ADDED

        self._window_idx_dag = nx.DiGraph() # dag representing dependencies between decoding windows, not insts
        self._instructions = instructions
        # for inst in instructions:
        #     print(inst)

        self.full_window_dag = full_window_dag
        self._per_window_poisoned: list[int] = []
        self._per_window_wasted_rounds: list[int] = []

    def get_dist_to_t_gate(self, task_idx):
        if self.full_window_dag is not None:
            dist = nx.single_source_shortest_path_length(self.full_window_dag, task_idx)
        else:
            dist = nx.single_source_shortest_path_length(self._window_idx_dag, task_idx)
        
        # print("dist ", dist, " start idx ", task_idx)
        for curr_task_idx, d in dist.items():
            curr_task = self._get_task_or_none(curr_task_idx)
            # if curr_task is None:
            #     print("curr task none ", curr_task_idx)
            if curr_task is not None:
                found = False
                # print(d)
                # TODO d seems to be incrementing?? I don't know why but can just use curr_task_inst_idx to get the value
                for curr_task_inst_idx in curr_task.window.parent_instr_idx:
                    # print("curr task indx ", curr_task_inst_idx, d, curr_task_idx, dist, self._instructions[curr_task_inst_idx].t_gate_bool)
                    if self._instructions[curr_task_inst_idx].t_gate_bool:
                        # print("curr task+d ", curr_task, " ", curr_task_idx, " ", d, " ", curr_task.window.parent_instr_idx)
                        # print("ACTUAL DISTANCE ", dist[curr_task_inst_idx])
                        # print("distance from t", d)
                        found = True
                        break

                if found:
                    break
        # print("found ", found)

    def step(self) -> list[int]:
        """Step decoding and speculation forward by one round (without creating
        any new processes)."""
        # Step decoders forward; check if any windows have completed
        # print("active window progress", self._active_window_progress)
        # print("active speculation progress", self._active_speculation_progress)

        for task_idx in self._active_window_progress: # iterate through all active tasks/windows
            self._active_window_progress[task_idx] -= 1 # decrease their # remaining rounds by 1
        completed_windows = []
        poisoned_speculations = []

        for task_idx, time_remaining in self._active_window_progress.items():
            # print("active window progress task ", self._get_task(task_idx))
            if time_remaining <= 0:
                completed_windows.append(task_idx) # if task's time remaining is <= 0, then we append this task index to completed windows

        self._num_completed_windows += len(completed_windows)
        for task_idx in self._topologically_sort(completed_windows): # topologicaclly sort all completed windows
            task = self._get_task(task_idx)
            task.decoding_completion_time = self._current_round
            self._active_window_progress.pop(task_idx) # task done, so no longer active
            task.completed_decoding = True # task done decoding
            self._decoded_unverified_tasks.add(task_idx) # task done decoding, but not yet verified

            # print("parent inst indx", task.window.parent_instr_idx)

            # dist = nx.single_source_shortest_path_length(self._window_idx_dag, task_idx)
            # print("dist ", dist, " start idx ", task_idx)
            # for curr_task_idx, d in dist.items():
            #     curr_task = self._get_task_or_none(curr_task_idx)
            #     if curr_task is None:
            #         print("curr task none ", curr_task_idx)
            #     else:
            #         found = False
            #         print(d)
            #         # TODO d seems to be incrementing?? I don't know why but can just use curr_task_inst_idx to get the value
            #         for curr_task_inst_idx in curr_task.window.parent_instr_idx:
            #             # print("curr task indx ", curr_task_inst_idx, d, curr_task_idx, dist, self._instructions[curr_task_inst_idx].t_gate_bool)
            #             if self._instructions[curr_task_inst_idx].t_gate_bool:
            #                 # print("curr task+d ", curr_task, " ", curr_task_idx, " ", d, " ", curr_task.window.parent_instr_idx)
            #                 # print("ACTUAL DISTANCE ", dist[curr_task_inst_idx])
            #                 found = True
            #                 break

            #         if found:
            #             break

            # for curr_task in nx.bfs_tree(self._window_idx_dag, task_idx):
            #     # This visits start_node and all reachable successors
            #     for curr_task_inst_idx in curr_task.window.parent_instr_idx:
            #         if self._instructions[curr_task_inst_idx].t_gate_bool

            # print(task.window.parent_instr_idx)
            # for instr_idx in task.window.parent_instr_idx:
            #     print("in for loop")
            #     print(self._instructions[instr_idx])
            #     print(self._instructions[instr_idx].t_gate_bool)
            #     print(self._instructions[instr_idx].duration)

                # print(Duration.get_true_duration(self._instructions[instr_idx].duration))
                # print(self._instructions[instr_idx].actual_duration_time)
                # if instr_idx in self._active_speculation_progress:
                #     print("active speculation", self._active_speculation_progress[instr_idx])

                # print("not fully decoded", self._not_fully_decoded_instructions)

            # Check if any speculations failed
            if self.speculation_mode: # if we turned speculation on
                for successor_idx in self._window_idx_dag.successors(task_idx): # get all successors of the current ask
                    successor = self._get_task_or_none(successor_idx) # get task of the successor
                    task = self._get_task(task_idx) # get current task
                    spec_acc_modifier = task.speculation_modifiers[successor_idx] if successor_idx in task.speculation_modifiers else 1.0 # get successor's speculation modifier if it exists
                    if successor and successor.decoding_start_time != -1: # if we have a successor task and it has started decoding
                        # if we miss-speculate, by multiplying misprediction accuracy by speculation accuracy modifier, and doing this to the exopnent of the number of faces this successor's window is touching our current window
                        # if self.rng.random() > (1-((1-self.speculation_accuracy)*spec_acc_modifier))**self._get_task(successor_idx).window.count_touching_faces(self._get_task(task_idx).window):
                        # print(task.window.speculation_accuracy)
                        # Currently put at DecodingWindow level -- needs to change a lot more places in both window_manager.py and window_builder.py -- but more fine granularity
                        # Can also just put at DecoderTask granularity, and edit one line in decoder_manager.py that creates DecoderTask, but is more coarse grain granularity
                        x = self.rng.random()
                        # print("self rng random ", x)
                        # print("task window spec acc", task.window.speculation_accuracy)
                        y = (1-((1-task.window.speculation_accuracy)*spec_acc_modifier))**self._get_task(successor_idx).window.count_touching_faces(self._get_task(task_idx).window)
                        # print("rhs if speculate ", y, " wind acc ", task.window.speculation_accuracy, " acc mod ", spec_acc_modifier, " touch face ", self._get_task(successor_idx).window.count_touching_faces(self._get_task(task_idx).window), " succ idx ", successor_idx, " task idx ", task_idx)
                        if x > y: # x > (1-((1-task.window.speculation_accuracy)*spec_acc_modifier))**self._get_task(successor_idx).window.count_touching_faces(self._get_task(task_idx).window): # instd of task.window_speculation_accuracy
                            # Missed speculation
                            # print("misspeculate", successor_idx)
                            self._num_failed_speculations += 1
                            assert successor.used_parent_speculations[task_idx] # asser that this successor did use the parent's (curr task_idx's) speculation result 
                            poisoned_speculations.append(successor_idx)

                            # print("task idx, poisoned speculation", task_idx, poisoned_speculations)
                            # update speculation modifiers (adjacent faces have
                            # higher failure rate)
                            poisoned_source_crs = successor.window.get_touching_commit_regions(task.window) # now that we know successor has been poisoned, get all of successor's commit regions that touch our curr tasks's commit regions. these are all the poisoned commit regions
                            for other_successor_idx in self._window_idx_dag.successors(successor_idx): # iterate through all successors of this successor in the window dag
                                other_successor = self._get_task_or_none(other_successor_idx) # get the task of this other successor
                                if other_successor:
                                    for other_cr in successor.window.get_touching_commit_regions(other_successor.window): 
                                        # if this other successor's commit region is touching our curr successor's commit region (which has been poisoned), we know that we need to modify the speculation accuracy of this other succ's commit rgn
                                        if any(cr.shares_edge(other_cr) for cr in poisoned_source_crs): # if shares a border with any of the poisoned commit regions
                                            successor.speculation_modifiers[other_successor_idx] = successor.speculation_modifiers.get(other_successor_idx, 1.0) * self.missed_speculation_modifier # multiply by missed speculation modifier (multiply existing missed speculation modifier if alr exists (from other missed speculation))
                                            # TODO: edge case where single
                                            # window has multiple adjacent
                                            # faces, and only some of them
                                            # should get the extra modifier. But
                                            # this is rare and our current
                                            # method is good enough for now.
                        else: # speculation did not fail
                            # Verify speculation
                            successor.used_parent_speculations[task_idx] = False # TODO why is this false? we did use the parent speculation, we just know we speculated correctly, right?
                            self._num_successful_speculations += 1

        # Step speculation forward
        if self.speculation_mode:
            for task_idx in self._active_speculation_progress:
                self._active_speculation_progress[task_idx] -= 1 # progress all speculations in progress by 1
            speculated_windows = []
            for task_idx, time_remaining in self._active_speculation_progress.items():
                if time_remaining <= 0:
                    speculated_windows.append(task_idx) # append to completed speculated windows if time remaining <= 0
            for task_idx in speculated_windows: # for every task in speculated windows, set completed speculation=True and it's no longer in active speculation progress
                task = self._get_task(task_idx)
                task.completed_speculation = True
                self._active_speculation_progress.pop(task_idx)

            # Update poisoned windows
            # For each poisoned window, reset any descendants that used the poisoned
            # speculation.
            all_poisoned_indices = []
            # print("poisoned speculations ", poisoned_speculations)
            for poisoned_task_idx in poisoned_speculations:
                poisoned_task = self._get_task_or_none(poisoned_task_idx)
                if poisoned_task and poisoned_task.decoding_start_time != -1: # if poisoned task has started decoding already
                    if self.poison_policy == 'descendants':
                        for descendant_idx in nx.descendants(self._window_idx_dag, poisoned_task_idx): # get all descendants of poisoned task
                            descendant = self._get_task_or_none(descendant_idx) # get task of descendatn
                            if descendant:
                                # print("descendant parent ", descendant.window.parent_instr_idx, poisoned_task.window.parent_instr_idx)
                                if descendant.decoding_start_time != -1: # if task started decoding, reset decoding task
                                    self._reset_decode_task(descendant_idx)
                                if descendant.speculation_start_time != -1: # if task started speculating, restart speculation task
                                    self._reset_speculate_task(descendant_idx)
                    elif self.poison_policy == 'successors':
                        if poisoned_task.completed_decoding:
                            # don't reset children, just mark them as used
                            # speculation rather than used decoding
                            all_poisoned_indices += self._poisoned_task_reset_children_that_used_decoding(poisoned_task_idx, only_mark_speculated=True)
                        # print("poisoned parent only ", poisoned_task.window.parent_instr_idx)
                        # self.get_dist_to_t_gate(poisoned_task_idx)
                        # for print_inst in poisoned_task.window.parent_instr_idx:
                        #     print("poisoned task insts ", print_inst, " ", self._instructions[print_inst])
                        # dist = nx.single_source_shortest_path_length(self._window_idx_dag, poisoned_task_idx)
                        # print("dist ", dist, " start idx ", poisoned_task_idx)
                        # for curr_task_idx, d in dist.items():
                        #     curr_task = self._get_task_or_none(curr_task_idx)
                        #     if curr_task is None:
                        #         print("curr task none ", curr_task_idx)
                        #     else:
                        #         found = False
                        #         print(d)
                        #         # TODO d seems to be incrementing?? I don't know why but can just use curr_task_inst_idx to get the value
                        #         for curr_task_inst_idx in curr_task.window.parent_instr_idx:
                        #             # print("curr task indx ", curr_task_inst_idx, d, curr_task_idx, dist, self._instructions[curr_task_inst_idx].t_gate_bool)
                        #             if self._instructions[curr_task_inst_idx].t_gate_bool:
                        #                 # print("curr task+d ", curr_task, " ", curr_task_idx, " ", d, " ", curr_task.window.parent_instr_idx)
                        #                 # print("ACTUAL DISTANCE ", dist[curr_task_inst_idx])
                        #                 print("distance from t", d)
                        #                 found = True
                        #                 break

                        #         if found:
                        #             break
                        self._reset_decode_task(poisoned_task_idx)
                    all_poisoned_indices.append(poisoned_task_idx)
            if self.lightweight_setting == 0:
                self._missed_speculation_events.append((self._current_round, all_poisoned_indices))

        self._current_round += 1
        # update num rounds where have only useless unwanted idle windows
        only_unwanted_idle = True
        for window in self._active_window_progress.keys():
            for parent_inst in self._tasks_by_idx[window].window.parent_instr_idx:
                if parent_inst != -1:
                    only_unwanted_idle = False
                    break
            if not only_unwanted_idle:
                break
        if only_unwanted_idle:
            self._unwanted_idle_rounds += 1
            
        if self.lightweight_setting == 0: # update bookeeping
            self._decode_processes_by_round.append(len(self._active_window_progress))
            self._speculate_processes_by_round.append(len(self._active_speculation_progress))
            self._completed_windows_by_round.append((self._completed_windows_by_round[-1] if self._current_round > 1 else 0) + len(completed_windows)) # add completed windows this round (cumulative sum returned)
        self._max_decoding_processes_used = max(self._max_decoding_processes_used, len(self._active_window_progress))
        self._max_speculation_processes_used = max(self._max_speculation_processes_used, len(self._active_speculation_progress))
        self._max_combined_processes_used = max(self._max_combined_processes_used, len(self._active_window_progress) + len(self._active_speculation_progress))
        self._decode_processor_spacetime_volume += len(self._active_window_progress) # num active windows decoding
        self._speculate_processor_spacetime_volume += len(self._active_speculation_progress) # num active windows speculating
        
        verified_tasks = set()
        for task_idx in self._decoded_unverified_tasks:
            task = self._get_task(task_idx)
            # if task did not use parent speculation, and all of its parents are verified
            # All parent speculations that this window used have now been confirmed (no longer speculative) aka: Have all my speculative parent dependencies been verified?
            # Every parent window itself has been verified.
            if all(val == False for val in task.used_parent_speculations.values()) and all(self._is_verified_task(parent_idx) for parent_idx in self._window_idx_dag.predecessors(task_idx)):
                verified_tasks |= self._verify_task_and_children(task_idx) # verify task and children, add to verified tasks
        self._decoded_unverified_tasks -= verified_tasks # subtract from unverified tasks list
        # print(verified_tasks)

        decoded_instructions = set()
        for instr_idx in self._not_fully_decoded_instructions:
            if all(self._is_verified_task(task_idx) for task_idx in self._instruction_tasks.get(instr_idx, set())): # check that all of this instruciotn's tasks are verified
                decoded_instructions.add(instr_idx) # if so, instruction is fully decoded
        self._not_fully_decoded_instructions -= decoded_instructions

        # print("decoder manager tasks by idx ", self._tasks_by_idx)
        
        return completed_windows

    def _is_verified_task(self, task_idx: int, treat_none_as_true: bool = False) -> bool:
        task = self._get_task_or_none(task_idx)
        if task: # task must be done decoding and task must not be in unverified task list
            return not (task.decoding_completion_time == -1 or task_idx in self._decoded_unverified_tasks)
        else:
            if treat_none_as_true:
                return True
            raise RuntimeError(f'Invalid task index: {task_idx}')

    def _verify_task_and_children(self, task_idx: int) -> set[int]:
        verified_tasks = set()
        if self._is_verified_task(task_idx): # if self is a verified task, just return empty set
            return verified_tasks
        verified_tasks.add(task_idx) # add curr task
        task = self._get_task(task_idx)
        assert all(val == False for val in task.used_parent_speculations.values())
        for instr_idx in task.window.parent_instr_idx: # iterate thru instructions that generated this window
            if instr_idx in self._instruction_unverified_task_counts: # if these insts are unverified
                self._instruction_unverified_task_counts[instr_idx] -= 1 # then we know one of its dependent tasks is now verified, so we -1
                if self._instruction_unverified_task_counts[instr_idx] == 0: # if there are no longer any unverified tasks that this inst is dependent on, we remove this task from the list
                    self._instruction_unverified_task_counts.pop(instr_idx)
        for child_idx in self._window_idx_dag.successors(task_idx): # get window successors of this task
            child = self._get_task_or_none(child_idx) # get the task of this child
            # if the child is done decoding and the child no longer is using any parent speculations and all window parents/predecessors are verified, then we know that we can verify this child as well
            if child and child.completed_decoding and all(val == False for val in child.used_parent_speculations.values()) and all(self._is_verified_task(parent_idx) for parent_idx in self._window_idx_dag.predecessors(child_idx)):
                verified_tasks |= self._verify_task_and_children(child_idx)

        return verified_tasks

    def _instruction_dag_descendants(self, instr_idx: int) -> set[int]: # get all descendants for seen instructions
        """Get descendants of an instruction in the schedule DAG, up to
        instructions that have begun decoding."""
        descendants = set()
        for child_idx in self.instruction_idx_dag.successors(instr_idx): # get the instruction dag, get successor instructions
            if child_idx in self._seen_instructions: # if we have seen this child instruction already, add this child to descendants and get the descendants of that instruction too
                descendants.add(child_idx)
                descendants |= self._instruction_dag_descendants(child_idx) # trying to get all descendants here (not just direct descendants)
        return descendants

    # topological order = linear ordering of its nodes such that every edge goes from an earlier node to a later node in the list.
    def _topologically_sort(self, task_indices): 
        """Topologically sort a subset of the window DAG."""
        if len(task_indices) == 0:
            return []
        # get subgraph containing all task indices and their descendants
        subgraph = nx.subgraph(self._window_idx_dag, set((itertools.chain.from_iterable(nx.descendants(self._window_idx_dag, idx) for idx in task_indices))) | set(task_indices))
        sorted = list(nx.topological_sort(subgraph)) # topological sort them using the window dag, which has dependencies btwn parent_window --> child_window (so all the parents come before children in this ordering)
        sorted = [idx for idx in sorted if idx in task_indices] # sort only the task_indices that we're given (only include indexes from the list of task_indices that we're given)
        return sorted

    # reset decoding task of task_idx
    def _reset_decode_task(self, task_idx):
        assert not self._is_verified_task(task_idx)
        task = self._get_task(task_idx)

        self._wasted_decode_volume += self._current_round - task.decoding_start_time # add to wasted decode volume
        self._num_discarded_decodes += 1
        if len(self._per_window_wasted_rounds) <= task.window_idx:
            self._per_window_wasted_rounds += [0] * (task.window_idx-len(self._per_window_wasted_rounds)+1)
        if len(self._per_window_poisoned) <= task.window_idx:
            self._per_window_poisoned += [0] * (task.window_idx-len(self._per_window_poisoned)+1)
        # print(task.window_idx)
        # print(len(self._per_window_wasted_rounds))
        self._per_window_wasted_rounds[task.window_idx] += self._current_round - task.decoding_start_time
        self._per_window_poisoned[task.window_idx] += 1
        # print("wasted decode volume ", self._current_round - task.decoding_start_time, " ", task_idx)

        task.decoding_start_time = -1
        if task_idx in self._active_window_progress: # if task is still actively decoding
            self._active_window_progress.pop(task_idx) # restart task's decoding
        else: # task already finished decoding
            task.decoding_completion_time = -1 # reset decoding completion time
            self._num_completed_windows -= 1 # remove from num_completed_windows
            task.completed_decoding = False # no longer done decoding
            self._decoded_unverified_tasks.remove(task_idx) # remove from unverified tasks too, bc need to restart decoding
        task.used_parent_speculations = {} # reset the used parent speculations too, bc completely restart decoding from verified point
        self._pending_decode_tasks.add(task_idx) # add task once again to pending decode tasks, bc need to restart decoding
        assert task_idx not in self._active_window_progress and task.decoding_start_time == -1 and task.decoding_completion_time == -1 and not task.completed_decoding # this task should not have started decoding at all
        return task_idx
    
    # reset speculation task of task_idx
    def _reset_speculate_task(self, task_idx):
        task = self._get_task(task_idx)
        # print("wasted speculation volume ", self._current_round - task.speculation_start_time)
        task.speculation_start_time = -1
        self._per_window_wasted_rounds[task.window_idx] += self._current_round - task.speculation_start_time
        self._per_window_poisoned[task.window_idx] += 1
        if task_idx in self._active_speculation_progress: # if task actively speculating
            self._active_speculation_progress.pop(task_idx)
        else: # if task finished speculating
            task.speculation_completion_time = -1
            task.completed_speculation = False
        task.speculation_modifiers = {} # reset, bc now we need to completely reset speculation from verified state
        # if ((task.window.speculation_accuracy > 0.7 and (not self._instructions[sorted(task.window.parent_instr_idx)[0]].t_gate_bool or self._instructions[sorted(task.window.parent_instr_idx)[0]].group_name == "COND_S")) 
        #     or (task.window.speculation_accuracy > 0.8 and self._instructions[sorted(task.window.parent_instr_idx)[0]].t_gate_bool and self._instructions[sorted(task.window.parent_instr_idx)[0]].group_name != "COND_S")): # 0.5  task.window.speculation_accuracy > 0
        #     self._pending_speculate_tasks.add(task_idx) # add anew to pending speculate tasks list

        self._pending_speculate_tasks.add(task_idx)

        # if ((task.window.speculation_accuracy > 0.8)): # 0.5  task.window.speculation_accuracy > 0
        #     self._pending_speculate_tasks.add(task_idx) # add anew to pending speculate tasks list
        assert task_idx not in self._active_speculation_progress and task.speculation_start_time == -1 and task.speculation_completion_time == -1 and not task.completed_speculation
        return task

    # basically, don't reset immediately -- just mark as used this parent's speculation, and later, if we find out this parent decoding was wrong (which we assume is same pr as missed spec), then we restart only then
    def _poisoned_task_reset_children_that_used_decoding(self, task_idx, only_mark_speculated=False):
        """Recursively reset children of a completed-and-then-poisoned task (if
        children used the completed decoding result).

        Args:
            task_idx: Index of the poisoned task.
            only_mark_speculated: If True, only mark children as having used the
                parent's speculation, rather than actually resetting them. This
                means that we may later have a chance of needing to redo them if
                we realize that the parent decoding is different from what it
                used to be (which we assume has the same probability as a missed
                speculation).
        """
        poisoned_indices = []
        for child_idx in self._window_idx_dag.successors(task_idx): # get children windows from window dag
            child = self._get_task_or_none(child_idx)
            if child and child.decoding_start_time != -1 and not child.used_parent_speculations[task_idx]: # if child started decoding and not used parent speculation of this task yet
                if only_mark_speculated:
                    child.used_parent_speculations[task_idx] = True # set to true for only_marked_speculated
                else: # iterate through all of child's descendants, and see if we have to reset them too
                    if child.completed_decoding: # if child is completed decoding, then we want to add this child's successors to poisoned indices 
                        poisoned_indices += self._poisoned_task_reset_children_that_used_decoding(child_idx) # recursively go through child now for this to get the indices of all of its successors too TODO why are we doing this for scuccessors policy? I guess find all the iwndows that were affected?
                    self._reset_decode_task(child_idx) # reset decode task of child that finished decoding because it's invalid now
        return poisoned_indices # return a list of all reset descendant tasks (not including self) (TODO but I'm pretty sure it's always empty)
    
    def get_speculation_depth_OLD(self, parent_idx, depth) -> int:
        task = self._get_task(parent_idx)
        parents = list(self._window_idx_dag.predecessors(task.window_idx))
        maxDepth = depth

        if all(self._completed_decoding(parent_idx) for parent_idx in parents):
            return depth
        else:
            for parent_idx in parents:
                if not self._completed_decoding(parent_idx) and self._completed_speculation(parent_idx):
                    depth1 = self.get_speculation_depth(parent_idx, depth+1)
                    if depth1 > maxDepth:
                        maxDepth = depth1

            return maxDepth

    def first_distance_where_all_flagged(self, start):
        visited = {start}
        frontier = {start}
        dist = -1 # 0

        while frontier:
            next_frontier = set()

            for node in frontier:
                for pred in self._window_idx_dag.predecessors(node):
                    if pred not in visited:
                        visited.add(pred)
                        next_frontier.add(pred)

            dist += 1

            if not next_frontier:
                #return None
                return dist

            if all(self._is_verified_task(n) for n in next_frontier):
                return dist

            frontier = next_frontier
        
    def get_speculation_depth(self, task_idx, next_tasks_to_decode) -> int:
        task = self._get_task(task_idx)

        if task_idx in self._pending_decode_tasks:
            parents = list(self._window_idx_dag.predecessors(task.window_idx))

            # no dependencies (regardless of speculated or not) passed to me yet, so I'm not even ready to decode yet even with speculated deps
            if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
                return
            
            depth = self.first_distance_where_all_flagged(task_idx)
            # if depth:
            next_tasks_to_decode[depth].append(task_idx)
            # else:
            #     print("NONE VERIFIED")

            # if all(self._is_verified_task(parent_idx) for parent_idx in parents):
            #     next_tasks_to_decode[depth].append(task_idx)
            #     return
            # else:
            #     parent_done = True
            #     for parent_idx in parents:
            #         parent_task = self._get_task(parent_idx)
            #         parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

            #         if not all(self._is_verified_task(parent_idx) for parent_idx in parents_parents):
            #             parent_done = False
            #             break

            #     if parent_done:
            #         next_tasks_to_decode[1].append(task_idx)
            #     else: # added only below
            #         parent_parent_done = True
            #         for parent_idx in parents:
            #             parent_task = self._get_task(parent_idx)
            #             parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

            #             for parent_parent_idx in parents_parents:
            #                 parent_parent_task = self._get_task(parent_parent_idx)
            #                 parents_parents_parents = list(self._window_idx_dag.predecessors(parent_parent_task.window_idx))

            #                 if not all(self._is_verified_task(parent_idx) for parent_idx in parents_parents_parents):
            #                     parent_parent_done = False
            #                     break

            #             if not parent_parent_done:
            #                 break

            #         if parent_parent_done:
            #             next_tasks_to_decode[2].append(task_idx)


    def update_decoding(self, new_windows: list[DecodingWindow], purged_indices: list[int], window_idx_dag: nx.DiGraph) -> None:
        """Update state of processing windows and start any new decoding
        processes, if possible.

        Currently, assumes that speculation is performed at the beginning of
        decoding a window. E.g. if speculation takes 2 rounds and decoding a
        window takes 10 rounds, the speculation will be completed 2 rounds after
        the decoder starts decoding the window.

        Args:
            all_windows: List of all decoding windows.
            window_idx_dag: Directed acyclic graph representing the dependencies
                between decoding windows.
        """
        # Check dependencies and start new speculation and decoding processes
        # print("pending decode tasks1 ", self._pending_decode_tasks)
        self._window_idx_dag = window_idx_dag # update decoding window dag to the new one (dag represents dependencies btwn decoding windows)
        new_task_indices = set(w.window_idx for w in new_windows) # get all new windows and their new task indices
        new_tasks = [DecoderTask(window=window, window_idx=window.window_idx) for window in new_windows] # create new decoder tasks for each of the new windows # TODO alternate way for window spec acc , window_speculation_accuracy=0.5
        self._pending_decode_tasks |= new_task_indices # add all of these new tasks to pending decode tasks
        if self.speculation_mode: # if we are speculating, then we also want to add the new task indices to the pending speculation tasks, since we want to speculate these
            # self._pending_speculate_tasks |= new_task_indices
            copy_new_task_indices = new_task_indices.copy()
            for task in new_tasks:
                # if ((task.window.speculation_accuracy > 0.7 and (not self._instructions[sorted(task.window.parent_instr_idx)[0]].t_gate_bool or self._instructions[sorted(task.window.parent_instr_idx)[0]].group_name == "COND_S")) 
                #     or (task.window.speculation_accuracy > 0.8 and self._instructions[sorted(task.window.parent_instr_idx)[0]].t_gate_bool and self._instructions[sorted(task.window.parent_instr_idx)[0]].group_name != "COND_S")) and task.window.window_idx in copy_new_task_indices: # .5
                #     self._pending_speculate_tasks.add(task.window.window_idx)
                #     copy_new_task_indices.remove(task.window.window_idx)
                # if ((task.window.speculation_accuracy > 0.8)) and task.window.window_idx in copy_new_task_indices: # .5
                #     self._pending_speculate_tasks.add(task.window.window_idx)
                #     copy_new_task_indices.remove(task.window.window_idx)
                if task.window.window_idx in copy_new_task_indices: # .5
                    self._pending_speculate_tasks.add(task.window.window_idx)
                    copy_new_task_indices.remove(task.window.window_idx)
        
        # print("pending decode tasks2 ", self._pending_decode_tasks)
        # get the length needed for the tasks_by_idx list, which stores the number of tasks per window index (so we want to get the maximum window index to get the length of this list)
        new_task_len = max(len(self._tasks_by_idx), max((task.window_idx for task in new_tasks), default=-1) + 1) # second clause finds the max window idx within new tasks, and adds 1 since indices are 0-based
        if len(self._tasks_by_idx) < new_task_len: # if our current tasks_by_idx list is less than new_task_len, add to it to get it to new_task_len
            self._tasks_by_idx += [None] * (new_task_len - len(self._tasks_by_idx))
            self._per_window_poisoned += [0] * (new_task_len - len(self._tasks_by_idx))
            self._per_window_wasted_rounds += [0] * (new_task_len - len(self._tasks_by_idx))
        for task in new_tasks: # iterate through every new task
            self._tasks_by_idx[task.window_idx] = task # get the task's window index and set it to the current task
            for instr_idx in task.window.parent_instr_idx: # look at this window's parent instruction indexes (insts that affect it)
                if instr_idx != -1:
                    self._instruction_unverified_task_counts[instr_idx] = self._instruction_unverified_task_counts.get(instr_idx, 0) + 1 # increment inst unverified task counts (per inst, how many unverified tasks they have)
                    self._not_fully_decoded_instructions.add(instr_idx) # add this inst to not fully decoded inst (bc it has a window that is not yet decoded)
                    self._instruction_tasks[instr_idx] = self._instruction_tasks.get(instr_idx, set()) | {task.window_idx} # add this new task to this instruction's set of tasks
                    self._seen_instructions.add(instr_idx) # add this inst to the seen instructions list
        unprocessed_task_indices = self._pending_decode_tasks | self._pending_speculate_tasks # get all unprocessed task indexes (include both decode and speculate tasks)
        # print("pending decode tasks3 ", self._pending_decode_tasks)

        next_tasks_to_decode = defaultdict(list)
        next_tasks = []
        # all(val == False for val in task.used_parent_speculations.values())
        # # CANNOT USE THIS BECAUSE AT THIS POINT TASK's USED_PARENT_SPECULATIONS IS NOT UPDATED YET!!!
        # if self.max_parallel_processes:
        #     # print(self._decoded_unverified_tasks)
        #     for task_idx in unprocessed_task_indices:
        #         if task_idx in self._active_window_progress:
        #             continue

        #         task = self._get_task(task_idx)

        #         # if len(next_tasks_to_decode) >= self.max_parallel_processes - len(self._active_window_progress):
        #         #     break

        #         if task_idx in self._pending_decode_tasks:
        #             parents = list(self._window_idx_dag.predecessors(task.window_idx))
        #             if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
        #                 continue

        #             if all(val == False for val in task.used_parent_speculations.values()):
        #                 next_tasks_to_decode[0].append(task_idx)
        #                 continue
        #             else:
        #                 parent_done = True
        #                 for parent_idx in parents:
        #                     parent_task = self._get_task(parent_idx)
        #                     parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

        #                     if not all(val == False for val in parent_task.used_parent_speculations.values()):
        #                         parent_done = False
        #                         break

        #                 if parent_done:
        #                     next_tasks_to_decode[1].append(task_idx)
        #                 else: # added only below
        #                     parent_parent_done = True
        #                     for parent_idx in parents:
        #                         parent_task = self._get_task(parent_idx)
        #                         parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

        #                         for parent_parent_idx in parents_parents:
        #                             parent_parent_task = self._get_task(parent_parent_idx)
        #                             parents_parents_parents = list(self._window_idx_dag.predecessors(parent_parent_task.window_idx))

        #                             if not all(val == False for val in parent_parent_task.used_parent_speculations.values()):
        #                                 parent_parent_done = False
        #                                 break

        #                         if not parent_parent_done:
        #                             break

        #                     if parent_parent_done:
        #                         next_tasks_to_decode[2].append(task_idx)

        #     if next_tasks_to_decode:
        #         for dist in sorted(next_tasks_to_decode):
        #             val = next_tasks_to_decode[dist]
        #             for task in val:
        #                 next_tasks.append(task)

        #                 if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
        #                     break
                    
        #             if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
        #                 break
        #         print(next_tasks_to_decode)
        #         print(next_tasks)

        # this is based on completed decoding, but here I'm pretty sure I care about completed verification not completed decoding (bc completed decoding stil not verified)
        # if self.max_parallel_processes:
        #     for task_idx in unprocessed_task_indices:
        #         if task_idx in self._active_window_progress:
        #             continue

        #         task = self._get_task(task_idx)

        #         # if len(next_tasks_to_decode) >= self.max_parallel_processes - len(self._active_window_progress):
        #         #     break

        #         if task_idx in self._pending_decode_tasks:
        #             parents = list(self._window_idx_dag.predecessors(task.window_idx))
        #             if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
        #                 continue

        #             if all(self._completed_decoding(parent_idx) for parent_idx in parents):
        #                 next_tasks_to_decode[0].append(task_idx)
        #                 continue
        #             else:
        #                 parent_done = True
        #                 for parent_idx in parents:
        #                     parent_task = self._get_task(parent_idx)
        #                     parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

        #                     if not all(self._completed_decoding(parent_idx) for parent_idx in parents_parents):
        #                         parent_done = False
        #                         break

        #                 if parent_done:
        #                     next_tasks_to_decode[1].append(task_idx)
        #                 else: # added only below
        #                     parent_parent_done = True
        #                     for parent_idx in parents:
        #                         parent_task = self._get_task(parent_idx)
        #                         parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

        #                         for parent_parent_idx in parents_parents:
        #                             parent_parent_task = self._get_task(parent_parent_idx)
        #                             parents_parents_parents = list(self._window_idx_dag.predecessors(parent_parent_task.window_idx))

        #                             if not all(self._completed_decoding(parent_idx) for parent_idx in parents_parents_parents):
        #                                 parent_parent_done = False
        #                                 break

        #                         if not parent_parent_done:
        #                             break

        #                     if parent_parent_done:
        #                         next_tasks_to_decode[2].append(task_idx)

        #     if next_tasks_to_decode:
        #         for dist in sorted(next_tasks_to_decode):
        #             val = next_tasks_to_decode[dist]
        #             for task in val:
        #                 next_tasks.append(task)

        #                 if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
        #                     break
                    
        #             if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
        #                 break
        #         print(next_tasks_to_decode)
        #         print(next_tasks)

        # self._is_verified_task(task_idx)
        # THIS ONE BELOW IS THE ONE -- after checking with verified tasks, this is the one that follows the expected pattern!!!   
        if self.max_parallel_processes:
            for task_idx in unprocessed_task_indices:
                # if task_idx in self._active_window_progress:
                #     continue

                self.get_speculation_depth(task_idx, next_tasks_to_decode)

                # task = self._get_task(task_idx)

                # # if len(next_tasks_to_decode) >= self.max_parallel_processes - len(self._active_window_progress):
                # #     break

                # if task_idx in self._pending_decode_tasks:
                #     parents = list(self._window_idx_dag.predecessors(task.window_idx))
                #     if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
                #         continue

                #     if all(self._is_verified_task(parent_idx) for parent_idx in parents):
                #         next_tasks_to_decode[0].append(task_idx)
                #         continue
                #     else:
                #         parent_done = True
                #         for parent_idx in parents:
                #             parent_task = self._get_task(parent_idx)
                #             parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

                #             if not all(self._is_verified_task(parent_idx) for parent_idx in parents_parents):
                #                 parent_done = False
                #                 break

                #         if parent_done:
                #             next_tasks_to_decode[1].append(task_idx)
                #         else: # added only below
                #             parent_parent_done = True
                #             for parent_idx in parents:
                #                 parent_task = self._get_task(parent_idx)
                #                 parents_parents = list(self._window_idx_dag.predecessors(parent_task.window_idx))

                #                 for parent_parent_idx in parents_parents:
                #                     parent_parent_task = self._get_task(parent_parent_idx)
                #                     parents_parents_parents = list(self._window_idx_dag.predecessors(parent_parent_task.window_idx))

                #                     if not all(self._is_verified_task(parent_idx) for parent_idx in parents_parents_parents):
                #                         parent_parent_done = False
                #                         break

                #                 if not parent_parent_done:
                #                     break

                #             if parent_parent_done:
                #                 next_tasks_to_decode[2].append(task_idx)

            if next_tasks_to_decode:
                for dist in sorted(next_tasks_to_decode): # , reverse=True
                    val = next_tasks_to_decode[dist]
                    for task in val:
                        next_tasks.append(task)

                        if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
                            break
                    
                    if len(next_tasks) >= self.max_parallel_processes - len(self._active_window_progress):
                        break
                # print(next_tasks_to_decode)
            

        # # TODO ADDED: initially iterate through unprocessed_task_indices and get the most parallel ones to launch based on max parallel processes
        # # this is restricting the NUMBER OF DECODERS, NOT the number of SPECULATORS
        # next_tasks_to_decode = []
        # print("b4 parallel processes new")
        # if self.max_parallel_processes:
        #     print("in max parallel procs")
        #     no_speculation_tasks = []
        #     speculation_dists = defaultdict(list)
        #     for task_idx in unprocessed_task_indices:
        #         task = self._get_task(task_idx)

        #         # if self.max_parallel_processes and len(next_tasks_to_decode) >= self.max_parallel_processes - len(self._active_window_progress):
        #         #     break
                
        #         if task_idx in self._pending_decode_tasks:
        #             parents = list(self._window_idx_dag.predecessors(task.window_idx))
        #             if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
        #                 continue
        #             # begin decoding
        #             assert not self._completed_decoding(task_idx) and task_idx not in self._active_window_progress
        #             print("after assert ever?")

                    

        #             # if this window's dependencies are NOT speculated ones, then we always definitely want to decode this window first
        #             if all(self._completed_decoding(parent_idx) for parent_idx in parents):
        #                 no_speculation_tasks.append(task_idx)
        #             else: # at least one of this window's deps is speculated
        #                 for parent_idx in parents:
        #                     if not self._completed_decoding(parent_idx) and self._completed_speculation(parent_idx):
        #                         depth = self.get_speculation_depth(parent_idx, 1)
        #                         speculation_dists[depth].append(parent_idx)

        #     print("before while next tasks to decode")
        #     print(no_speculation_tasks)
        #     print(speculation_dists)
        #     print(self.max_parallel_processes - len(self._active_window_progress))
        #     while len(next_tasks_to_decode) < self.max_parallel_processes - len(self._active_window_progress) and len(no_speculation_tasks) > 0 and speculation_dists:
        #         # print(next_tasks_to_decode)
        #         if len(no_speculation_tasks) > 0:
        #             next_tasks_to_decode.append(no_speculation_tasks[0])
        #             no_speculation_tasks.pop(0)

        #         if speculation_dists:
        #             smallest_key = min(speculation_dists)
        #             if speculation_dists[smallest_key]:
        #                 first_val = speculation_dists[smallest_key][0]
        #                 next_tasks_to_decode.append(first_val)
        #                 speculation_dists[smallest_key].pop(0)
        #                 if len(speculation_dists[smallest_key]) == 0:
        #                     del speculation_dists[smallest_key]

        #         # if this window's dependences ARE speculated ones, then we want to reconsider whether to decode this window or another one that has less deep speculation
                
        # print("after parallel processes new")
        # print(next_tasks_to_decode)

        while len(unprocessed_task_indices) > 0: # process tasks
            task_idx = unprocessed_task_indices.pop()
            task = self._get_task(task_idx)
            assert task.window_idx == task_idx # the window index should be the same as the task index

            if task.completed_decoding or not task.window.constructed: # if the task is completed decoding or the window is not constructed, but we mark it as unprocessed, there's an error
                raise RuntimeError(f'Task is complete, but marked as unprocessed: {task_idx}')

            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes: # if we have more active windows in progress than our maximum parallel process limit, we exit
                break

            # begin a speculation
            # separate speculation mode here!, task is pending speculated and has not completed decoding, then we want to begin speculation
            if task_idx in self._pending_speculate_tasks and self.speculation_mode == 'separate' and not self._completed_decoding(task_idx):
                assert task_idx not in self._active_speculation_progress
                self._pending_speculate_tasks.remove(task_idx) # we start speculation for this task, so remove from pending list
                task.speculation_start_time = self._current_round # start time for this speculation is the current round
                if task.window.speculation_time > 0: # if self.speculation_time > 0: # speculation time is teh number of rounds needed for speculating this specific window
                    self._active_speculation_progress[task_idx] = task.window.speculation_time # self.speculation_time # set the active speculation progress for this task index to this time (the list holds the remaining number of rounds until this task's speculation is complete)
                else:
                    task.completed_speculation = True # if speculation time = 0, then we know speculation has completed
                    task.speculation_completion_time = self._current_round
                    # since we've speculated this task, we can put its successors in unprocessed (but are ready to start processing) task indices array. these successors should not be done decoding and also should have a task
                    unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
            
            # if the task index is in the pending decode tasks array
            # print("pending decode tasks ", self._pending_decode_tasks)
            if self.max_parallel_processes:
                if task_idx in self._pending_decode_tasks: #  and task_idx in next_tasks_to_decode
                    # TODO technically we would need to first calculate how many next tasks we have then if we don't have enough
                    # we would want to grab from the front instead of us grabbing from the back in case the back doesn't have 
                    # enough tasks to fill our load
                    if len(next_tasks) > 0:
                        if not (len(self._active_window_progress) + len(next_tasks) < self.max_parallel_processes):
                            if task_idx not in next_tasks:
                                continue
                            else:
                                next_tasks.remove(task_idx)
                        else:
                            if task_idx in next_tasks:
                                next_tasks.remove(task_idx)
                    # window has not been processed yet
                    parents = list(self._window_idx_dag.predecessors(task.window_idx)) # get parent windows
                    # print("parents pending decoding ", parents)
                    # if any of the parents have not completed decoding or completed speculation, then we can't decode yet, since we haven't received our (actual or predicted) dependency bits
                    if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
                        # print("IN THIS BREAK4")
                        continue
                    # begin decoding
                    assert not self._completed_decoding(task_idx) and task_idx not in self._active_window_progress # assert this window has not started decoding yet and has not completed
                    self._pending_decode_tasks.remove(task_idx)
                    task.decoding_start_time = self._current_round
                    x = task.window.decoding_time_fn(task.window.total_spacetime_volume()) # self.decoding_time_function(task.window.total_spacetime_volume()) 
                    if len(self._per_window_wasted_rounds) <= task.window_idx:
                        self._per_window_wasted_rounds += [0] * (task.window_idx-len(self._per_window_wasted_rounds)+1)
                    if len(self._per_window_poisoned) <= task.window_idx:
                        self._per_window_poisoned += [0] * (task.window_idx-len(self._per_window_poisoned)+1)
                    # print("decoding time fcn result ", x)
                    # print(task_idx)
                    self._active_window_progress[task_idx] = x # self.decoding_time_function(task.window.total_spacetime_volume()) # set the remaining decoding time for this task (which rn, is the total decoding time for this window)
                    task.used_parent_speculations = {}
                    for parent_idx in parents: # check for each parent, whether we're using their speculated result or their actual verified result
                        parent = self._get_task(parent_idx)
                        if parent.completed_decoding:
                            task.used_parent_speculations[parent_idx] = False
                        else:
                            assert parent.completed_speculation
                            task.used_parent_speculations[parent_idx] = True
                    # if we're in integrated mode for speculation, we want to begin the speculation WITH the decoding (unlike separate mode, whose speculation was started above)
                    if self.speculation_mode == 'integrated' and task_idx in self._pending_speculate_tasks: # if our task is in the pending speculation tasks
                        # begin a speculation along with decoding
                        assert task_idx not in self._active_speculation_progress # has not started speculating
                        self._pending_speculate_tasks.remove(task_idx)
                        task.speculation_start_time = self._current_round
                        # handle speculation time
                        if task.window.speculation_time > 0: # if self.speculation_time > 0:
                            self._active_speculation_progress[task_idx] = task.window.speculation_time # self.speculation_time
                        else:
                            self._get_task(task_idx).completed_speculation = True
                            # add to unprocessed (but rdy to start processing) task indices all of this window's successors (which depended on this window), if these successors have a task and have not completed decoding
                            # TODO what if these windows have multiple dependencies
                            unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
            else:
                if task_idx in self._pending_decode_tasks:
                    # window has not been processed yet
                    parents = list(self._window_idx_dag.predecessors(task.window_idx)) # get parent windows
                    # print("parents pending decoding ", parents)
                    # if any of the parents have not completed decoding or completed speculation, then we can't decode yet, since we haven't received our (actual or predicted) dependency bits
                    if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents): 
                        # print("IN THIS BREAK4")
                        continue
                    # begin decoding
                    assert not self._completed_decoding(task_idx) and task_idx not in self._active_window_progress # assert this window has not started decoding yet and has not completed
                    self._pending_decode_tasks.remove(task_idx)
                    task.decoding_start_time = self._current_round
                    x = task.window.decoding_time_fn(task.window.total_spacetime_volume()) # self.decoding_time_function(task.window.total_spacetime_volume()) 
                    if len(self._per_window_wasted_rounds) <= task.window_idx:
                        self._per_window_wasted_rounds += [0] * (task.window_idx-len(self._per_window_wasted_rounds)+1)
                    if len(self._per_window_poisoned) <= task.window_idx:
                        self._per_window_poisoned += [0] * (task.window_idx-len(self._per_window_poisoned)+1)
                    # print("decoding time fcn result ", x)
                    # print(task_idx)
                    self._active_window_progress[task_idx] = x # self.decoding_time_function(task.window.total_spacetime_volume()) # set the remaining decoding time for this task (which rn, is the total decoding time for this window)
                    task.used_parent_speculations = {}
                    for parent_idx in parents: # check for each parent, whether we're using their speculated result or their actual verified result
                        parent = self._get_task(parent_idx)
                        if parent.completed_decoding:
                            task.used_parent_speculations[parent_idx] = False
                        else:
                            assert parent.completed_speculation
                            task.used_parent_speculations[parent_idx] = True
                    # if we're in integrated mode for speculation, we want to begin the speculation WITH the decoding (unlike separate mode, whose speculation was started above)
                    if self.speculation_mode == 'integrated' and task_idx in self._pending_speculate_tasks: # if our task is in the pending speculation tasks
                        # begin a speculation along with decoding
                        assert task_idx not in self._active_speculation_progress # has not started speculating
                        self._pending_speculate_tasks.remove(task_idx)
                        task.speculation_start_time = self._current_round
                        # handle speculation time
                        if task.window.speculation_time > 0: # if self.speculation_time > 0:
                            self._active_speculation_progress[task_idx] = task.window.speculation_time # self.speculation_time
                        else:
                            self._get_task(task_idx).completed_speculation = True
                            # add to unprocessed (but rdy to start processing) task indices all of this window's successors (which depended on this window), if these successors have a task and have not completed decoding
                            # TODO what if these windows have multiple dependencies
                            unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
        # print(self._active_window_progress)


    # check if a task is done completed decoding or not
    def _completed_decoding(self, task_idx: int) -> bool:
        task = self._get_task_or_none(task_idx)
        if task:
            return task.completed_decoding
        return False
    
    # check if a task is completed speculation or not
    def _completed_speculation(self, task_idx: int) -> bool:
        task = self._get_task_or_none(task_idx)
        if task:
            return task.completed_speculation
        return False

    # get the task of a task idx; if not exist error out
    def _get_task(self, task_idx: int) -> DecoderTask:
        if task_idx >= len(self._tasks_by_idx):
            raise RuntimeError(f'Invalid task index: {task_idx}')
        task = self._tasks_by_idx[task_idx]
        if task is None:
            raise RuntimeError(f'Invalid or deleted task index: {task_idx}')
        else:
            return task
    
    # get task of task idx; if not exist return None
    def _get_task_or_none(self, task_idx: int) -> DecoderTask | None:
        if task_idx >= len(self._tasks_by_idx):
            return None
        task = self._tasks_by_idx[task_idx]
        if task is None:
            return None
        return task
    
    # get all incomplete/unverified instruction indices along with all their descendants, to get the entire set of instruction indexes that have not been decoded
    def get_incomplete_instruction_indices(self) -> set[int]:
        """Return the set of instruction idx that have not been decoded."""
        # return set(self._instruction_unverified_task_counts.keys())
        return set(itertools.chain.from_iterable(self._instruction_dag_descendants(instr_idx) for instr_idx in self._instruction_unverified_task_counts.keys() if instr_idx != -1)) | self._instruction_unverified_task_counts.keys()

    # get metadata depending on our lightweight setting
    def get_data(self) -> DecoderData:
        per_parent_inst = defaultdict(list)

        for task_idx, task in enumerate(self._tasks_by_idx):
            if task:
                per_parent_inst[task.window.parent_instr_idx].append(task_idx)

        per_parent_inst = dict(per_parent_inst)

        if self.lightweight_setting == 0: # returns everything
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_decoders=self._max_decoding_processes_used,
                max_parallel_speculators=self._max_speculation_processes_used,
                max_parallel_combined_processes=self._max_combined_processes_used,
                decode_process_volume=self._decode_processor_spacetime_volume,
                speculate_process_volume=self._speculate_processor_spacetime_volume,
                num_completed_windows=self._num_completed_windows,
                decode_processes_by_round=self._decode_processes_by_round,
                speculate_processes_by_round=self._speculate_processes_by_round,
                completed_windows_by_round=self._completed_windows_by_round,
                window_speculation_start_times={task_idx:task.speculation_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_start_times={task_idx:task.decoding_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_completion_times={task_idx:task.decoding_completion_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                missed_speculation_events=self._missed_speculation_events,
                num_failed_speculations=self._num_failed_speculations,
                num_discarded_decodes=self._num_discarded_decodes,
                wasted_decode_volume=self._wasted_decode_volume,
                num_successful_speculations=self._num_successful_speculations,
                per_window_wasted_rounds=self._per_window_wasted_rounds,
                per_window_poisoned=self._per_window_poisoned,
                per_window_parent_inst={task_idx:task.window.parent_instr_idx for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_window_spec_acc={task_idx:round(task.window.speculation_accuracy, 2) for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_inst_windows=per_parent_inst,
            )
        elif self.lightweight_setting == 1: # return less
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_decoders=self._max_decoding_processes_used,
                max_parallel_speculators=self._max_speculation_processes_used,
                max_parallel_combined_processes=self._max_combined_processes_used,
                decode_process_volume=self._decode_processor_spacetime_volume,
                speculate_process_volume=self._speculate_processor_spacetime_volume,
                num_completed_windows=self._num_completed_windows,
                decode_processes_by_round=None,
                speculate_processes_by_round=None,
                completed_windows_by_round=None,
                window_speculation_start_times={task_idx:task.speculation_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_start_times={task_idx:task.decoding_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_completion_times={task_idx:task.decoding_completion_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                missed_speculation_events=self._missed_speculation_events,
                num_failed_speculations=self._num_failed_speculations,
                num_discarded_decodes=self._num_discarded_decodes,
                wasted_decode_volume=self._wasted_decode_volume,
                num_successful_speculations=self._num_successful_speculations,
                per_window_wasted_rounds=self._per_window_wasted_rounds,
                per_window_poisoned=self._per_window_poisoned,
                per_window_parent_inst={task_idx:task.window.parent_instr_idx for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_window_spec_acc={task_idx:round(task.window.speculation_accuracy, 2) for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_inst_windows=per_parent_inst,
            )
        elif self.lightweight_setting == 2 or self.lightweight_setting == 3: # return even less
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_decoders=self._max_decoding_processes_used,
                max_parallel_speculators=self._max_speculation_processes_used,
                max_parallel_combined_processes=self._max_combined_processes_used,
                decode_process_volume=self._decode_processor_spacetime_volume,
                speculate_process_volume=self._speculate_processor_spacetime_volume,
                num_completed_windows=self._num_completed_windows,
                decode_processes_by_round=None,
                speculate_processes_by_round=None,
                completed_windows_by_round=None,
                window_speculation_start_times=None,
                window_decoding_start_times=None,
                window_decoding_completion_times=None,
                missed_speculation_events=None,
                num_failed_speculations=self._num_failed_speculations,
                num_discarded_decodes=self._num_discarded_decodes,
                wasted_decode_volume=self._wasted_decode_volume,
                num_successful_speculations=self._num_successful_speculations,
                per_window_wasted_rounds=self._per_window_wasted_rounds,
                per_window_poisoned=self._per_window_poisoned,
                per_window_parent_inst={task_idx:task.window.parent_instr_idx for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_window_spec_acc={task_idx:round(task.window.speculation_accuracy, 2) for task_idx,task in enumerate(self._tasks_by_idx) if task},
                per_inst_windows=per_parent_inst,
            )
        else:
            raise RuntimeError('Invalid lightweight setting')