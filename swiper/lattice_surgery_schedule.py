from dataclasses import dataclass, field
import networkx as nx
import numpy as np
from enum import Enum
import copy

class Duration(Enum):
    D = 1
    HALF_D = 2
    HALF_D_PLUS_2 = 3

    def __str__(self):
        if self == Duration.D:
            return 'D'
        elif self == Duration.HALF_D:
            return 'HALF_D'
        elif self == Duration.HALF_D_PLUS_2:
            return 'HALF_D_PLUS_2'
        
    @classmethod
    def from_str(cls, s):
        if s == 'D':
            return cls.D
        elif s == 'HALF_D':
            return cls.HALF_D
        elif s == 'HALF_D_PLUS_2':
            return cls.HALF_D_PLUS_2
        elif s.isdigit():
            return int(s)
        else:
            raise ValueError(f'Invalid duration string: {s}')

    # we're returning multiples of distance in this function
    # convert duration to an actual integer, based on the distance and what the duration "command" is
    @staticmethod
    def get_true_duration(duration: 'Duration | int', distance: int):
        if isinstance(duration, Duration):
            if duration == Duration.D:
                return distance
            elif duration == Duration.HALF_D:
                return distance // 2
            elif duration == Duration.HALF_D_PLUS_2:
                return distance // 2 + 2
        return duration

# An Instruction represents a single lattice surgery operation
# frozenset = immutable set
# default_factory tells Python to call frozenset() each time a new object is created to produce a fresh empty frozenset as the default value
@dataclass(frozen=True)
class Instruction:
    name: str # type of lattice surgery op (e.g. MERGE, DISCARD)
    idx: int # index of instruction (its position in the schedule)
    patches: frozenset[tuple[int, int]] # the logical qubit patches that this instruction acts on
    duration: Duration | int # the time this operation takes, in rounds (can be of type Duration or int)
    conditioned_on_idx: frozenset[int] = field(default_factory=frozenset) # indices of instructions whose outcomes this one depends on logically (for conditional operations)
    conditional_dependencies: frozenset[int] = field(default_factory=frozenset) # which later instructions depend on this one's outcome
    conditioned_on_completion_idx: frozenset[int] = field(default_factory=frozenset) # operations that must finish before this one can be considered complete
    conditional_completion_dependencies: frozenset[int] = field(default_factory=frozenset) # these instructions depend on me finishing first
    merge_faces: frozenset[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=frozenset) # faces (shared boundaries) btwn patches being merged
    group_instr_indices: frozenset[int] = field(default_factory=frozenset) # when a logical gate consists of multiple lower-level insts (e.g. S gate is merge+measure+...), this groups them together
    group_name: str = '' # name of the instruction group
    t_gate_bool: bool = False # ADDED: whether the instruction is part of a t-gate group or not
    actual_duration_time: int = None # ADDED: get the actual duration time (want to know if a Y_MEAS duration is actually 0)

    # simply only replace the name
    def rename(self, new_name) -> 'Instruction':
        return Instruction(
            name=new_name,
            idx=self.idx,
            patches=self.patches,
            duration=self.duration,
            conditioned_on_idx=self.conditioned_on_idx,
            conditional_dependencies=self.conditional_dependencies,
            conditioned_on_completion_idx=self.conditioned_on_completion_idx,
            conditional_completion_dependencies=self.conditional_completion_dependencies,
            merge_faces=self.merge_faces,
            group_instr_indices=self.group_instr_indices,
            t_gate_bool=self.t_gate_bool,
        )
    
    def __str__(self): # str(list(x).replace(" ","")) makes a string of a list that looks like: [1,2,3]
        return f'{self.name} {self.idx} {str(list(self.patches)).replace(" ", "")} {self.duration} {str(list(self.conditioned_on_idx)).replace(" ", "")} {str(list(self.conditional_dependencies)).replace(" ", "")} {str(list(self.conditioned_on_completion_idx)).replace(" ", "")} {str(list(self.conditional_completion_dependencies)).replace(" ", "")} {str(list(self.merge_faces)).replace(" ", "")} {str(list(self.group_instr_indices)).replace(" ", "")} {self.group_name if self.group_name else "none"} {self.t_gate_bool} {self.actual_duration_time if self.actual_duration_time else "none"}'

    # convert from string to an actual Instruction. string is in format shown above from def __str__ method
    @classmethod
    def from_str(cls, s):
        s = s.split() # split by space
        return Instruction(
            name=s[0],
            idx=int(s[1]),
            patches=frozenset(eval(s[2])),
            duration=Duration.from_str(s[3]),
            conditioned_on_idx=frozenset(eval(s[4])),
            conditional_dependencies=frozenset(eval(s[5])),
            conditioned_on_completion_idx=frozenset(eval(s[6])),
            conditional_completion_dependencies=frozenset(eval(s[7])),
            merge_faces=frozenset(eval(s[8])),
            group_instr_indices=frozenset(eval(s[9])),
            group_name=s[10] if s[10] != 'none' else '',
        )
    
    # get total spacetime volume, which multiplies the area of the patch from the time as well (which is based on the distance) (?)
    def spacetime_volume(self, distance: int) -> int:
        return len(self.patches) * Duration.get_true_duration(self.duration, distance)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    instructions: list[Instruction] # list of lattice surgery operations. it's the schedule. the order is the order of execution
    _instructions_by_patch: dict[tuple[int, int], list[int]] # tracks all the instructions that have touched a given patch over time

    def __init__(self, generate_dag_incrementally: bool = True): # generate an empty schedule (list of instructions) and an empty directed graph
        self.instructions: list[Instruction] = []
        self._instructions_by_patch = {}
        self.generate_dag_incrementally = generate_dag_incrementally
        self._generated_dag = nx.DiGraph()

    def __len__(self): # length of the number of instructions
        return len(self.instructions)
    
    def __str__(self): # return a string of all the instructions
        return '\n'.join(str(instr) for instr in self.full_schedule().instructions)
    
    def __eq__(self, other): # entire list of instructions/schedule is the exact same for 2
        return self.full_schedule().instructions == other.full_schedule().instructions
    
    # copy
    def __copy__(self): # copies instructions, instructions by patch, and dag, and whether it generates dag incrementally
        instance = LatticeSurgerySchedule(generate_dag_incrementally=self.generate_dag_incrementally)
        instance.instructions = copy.deepcopy(self.instructions)
        instance._instructions_by_patch = copy.deepcopy(self._instructions_by_patch)
        instance._generated_dag = copy.deepcopy(self._generated_dag)
        return instance
    
    # create a new instance of the LatticeSurgerySchedule class from a string, which contains a schedule of lattice surgery instructions
    @classmethod
    def from_str(cls, s, generate_dag_incrementally: bool = False):
        instance = cls(generate_dag_incrementally=generate_dag_incrementally) # creates new instance of class by calling its constructor
        for line in s.split('\n'): # convert the string into a schedule of instructions in the new class instance
            instance._add_instruction(Instruction.from_str(line))
        return instance

    def full_schedule(self) -> 'LatticeSurgerySchedule':
        """Return a list of all instructions, with all the necessary DISCARDS
        placed at the end.
        """
        full_sched = self.__copy__()
        full_sched.discard(full_sched._get_final_active_patches()) # adds DISCARD insts to the lattice surgery schedule, marking the logical qubit patches that are no longer active
        return full_sched

    # initializes a logical ancilla patch in a T state
    def inject_T(self, patches: list[tuple[int, int]], t_gate_bool: bool=False):
        # tries to do an INJECT_T operation on all patches that are not already active
        for patch in patches:
            # if patch is already active, can't inject T gate
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name != 'DISCARD':
                raise ValueError(f'Tried to inject T gate on patch {patch} at instruction {len(self.instructions)}, but it was already active. If this was intended, make sure to DISCARD the patch first.')
            # injects T gate into this specific patch
            instruction = Instruction('INJECT_T', len(self.instructions), frozenset([patch]), Duration.D, t_gate_bool=t_gate_bool) # 1:name, 2:idx, 3:patches, 4:duration
            self._add_instruction(instruction) # add instruction to the current schedule

    # define a Y_measurement -- this is part of the conditional measurement in T-gate teleportation
    def Y_meas(self, patch_coords: tuple[int, int], conditioned_on_idx: int=None, t_gate_bool: bool=False):
        instruction = Instruction(
            name='Y_MEAS',
            idx=len(self.instructions),
            patches=frozenset([patch_coords]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_name=('Y_MEAS' if not conditioned_on_idx else 'CONDITIONAL_S'),
            t_gate_bool=t_gate_bool
        )
        self._add_instruction(instruction)
        
        # we update the inst that our curr inst is dependent on, and add our curr inst to that inst's conditional dependencies list (list of which insts are dependent on this inst)
        update_instr = self.instructions[conditioned_on_idx]
        self.instructions[conditioned_on_idx] = Instruction(
            update_instr.name,
            update_instr.idx,
            update_instr.patches,
            update_instr.duration,
            update_instr.conditioned_on_idx,
            update_instr.conditional_dependencies | frozenset([len(self.instructions) - 1]),
            update_instr.conditioned_on_completion_idx,
            update_instr.conditional_completion_dependencies,
            update_instr.merge_faces,
            update_instr.group_instr_indices,
            update_instr.group_name,
            update_instr.t_gate_bool,
        )
    
    # target patch is the data qubit that shld receive the logical S correction
    # ancilla patch is the ancilla qubit measured to assist with the correction -- it must not already be active
    def S(self, target_patch: tuple[int, int], ancilla_patch: tuple[int, int], conditioned_on_idx: int | None = None, t_gate_bool: bool=False):
        # Cannot initialize an ancilla patch for a patch that is already active
        if ancilla_patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[ancilla_patch][-1]].name != 'DISCARD': # check that the last inst in this patch's list of insts is not a DISCARD (if it were a DISCARD, taht would be fine)
            raise ValueError(f'Tried to initialize ancilla patch {ancilla_patch} for S instruction {len(self.instructions)}, but it is already active. If this was intended, make sure to DISCARD the patch first.')
        group_instr_indices = frozenset([len(self.instructions), len(self.instructions) + 1, len(self.instructions) + 2, len(self.instructions) + 3]) # next 4 insts will be part of this group of S-gate insts
        # create the group of instructions needed for conditional S 
        merge_instr = Instruction(
            name='MERGE',
            idx=len(self.instructions),
            patches=frozenset([target_patch, ancilla_patch]),
            duration=Duration.D,
            merge_faces=frozenset([(target_patch, ancilla_patch)]),
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
            t_gate_bool=t_gate_bool,
        )
        # we'll be measuring the ancilla
        y_cube_instr = Instruction(
            name='Y_MEAS',
            idx=len(self.instructions) + 1,
            patches=frozenset([ancilla_patch]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
            t_gate_bool=t_gate_bool,
        )
        idle = Instruction(
            name='IDLE',
            idx=len(self.instructions) + 2,
            patches=frozenset([target_patch]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
            t_gate_bool=t_gate_bool,
        )
        # we'll discard the ancilla afterwards
        discard = Instruction(
            name='DISCARD',
            idx=len(self.instructions) + 3,
            patches=frozenset([ancilla_patch]),
            duration=0,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
            t_gate_bool=t_gate_bool,
        )
        # if we're conditioned on an index, then we want to update the instruction that we're conditioned on (add our curr insts to the list of that inst that keeps track of what insts are conditioned on it)
        if conditioned_on_idx:
            update_instr = self.instructions[conditioned_on_idx]
            self.instructions[conditioned_on_idx] = Instruction(
                update_instr.name,
                update_instr.idx,
                update_instr.patches,
                update_instr.duration,
                update_instr.conditioned_on_idx,
                update_instr.conditional_dependencies | group_instr_indices,
                update_instr.conditioned_on_completion_idx,
                update_instr.conditional_completion_dependencies,
                update_instr.merge_faces,
                update_instr.group_instr_indices,
                update_instr.group_name,
                update_instr.t_gate_bool,
            )
        self._add_instruction(merge_instr)
        self._add_instruction(y_cube_instr)
        self._add_instruction(idle)
        self._add_instruction(discard)

                                              
    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]] = [],
            merge_faces: set[tuple[tuple[int, int]]] | None = None,
            duration: Duration | int = Duration.D,
            t_gate_bool: bool=False,
        ) -> int:
        """Lattice surgery merge-and-split operation involving two or more
        logical qubits.
        
        Args:
            active_qubits: List of logical qubits (patch coords) to merge.
            routing_qubits: List of routing patches that connect the active
                qubits. Must not already be active patches.
            merge_faces: If provided, a set of tuples of tuples, where each
                tuple contains two patch coordinates that share a merged
                boundary. If not provided, the merge faces are inferred from
                the active and routing qubits.
            duration: Duration of the merge operation. After the merge, the
                routing patches are discarded.
        
        Returns:
            The index of the merge instruction in the schedule.
        """
        # if more than 2 active patches, then need a routing patch (bc now the active patches can be connected in more than one way)
        if len(routing_qubits) == 0 and len(active_qubits) != 2:
            raise ValueError(f'No routing patches provided for merge instruction {len(self.instructions)}, but more than two active patches {active_qubits}.')
        # need routing patch provided also if the qubits are not directly adjacent to each other (adjacent means they are just a dist 1 away from each other)
        if len(routing_qubits) == 0 and np.linalg.norm(np.array(active_qubits[0]) - np.array(active_qubits[1])) != 1:
            raise ValueError(f'No routing patches provided for merge instruction {len(self.instructions)}, but active patches {active_qubits} are not adjacent.')
        # if merge_faces not providd, create them
        if not merge_faces:
            merge_faces = set()
            if len(routing_qubits) > 0:
                for qubit in active_qubits:
                    found_match = False
                    for routing in routing_qubits:
                        if np.linalg.norm(np.array(qubit) - np.array(routing)) == 1: # find the routing qubit that is directly adjacent to the current active qubit
                            if found_match:
                                raise ValueError('Multiple connections to routing space for one logical qubit; can\'t handle this case.')
                            found_match = True
                            merge_faces.add((qubit, routing)) # add these adjacent qubits to merge_faces (list of 2 patch coords that share a merged boundary)
                for i,routing_1 in enumerate(routing_qubits): # if there are adjacent routing qubits, then add them to merge_faces too, since they share a merged boundary
                    for routing_2 in routing_qubits[:i]:
                        if np.linalg.norm(np.array(routing_1) - np.array(routing_2)) == 1: # if 2 routing qubits are adajacent
                            merge_faces.add((routing_1, routing_2))
            else:
                if len(active_qubits) == 2: # if no routing qubits, just add the active_qubits to merge_faces directly, where we have 2 active_qubits
                    merge_faces.add(tuple(active_qubits))
                else:
                    raise ValueError('Can only merge two patches without routing patches.')
        for patch in active_qubits + routing_qubits: # for every patch, verify that no patch participates in more than 4 merge faces (for geometric consistency)
            assert sum(patch in face for face in merge_faces) <= 4, (patch, merge_faces)
        for patch in routing_qubits: # if initialize routing patch on a patch that is already active, error out
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name != 'DISCARD':
                raise ValueError(f'Tried to initialize routing patch {patch} for merge instruction {len(self.instructions)}, but it is already active. If this was intended, make sure to DISCARD the patch first.')
        # create merge instruction, and discard routing qubits (add discard instruction at the end)
        instruction = Instruction(
            name='MERGE',
            idx=len(self.instructions),
            patches=frozenset(active_qubits + routing_qubits),
            duration=duration,
            merge_faces=frozenset(merge_faces),
            t_gate_bool=t_gate_bool,
        )
        print("idx inst", instruction.idx)
        print(instruction.t_gate_bool, " t_gate_bool")
        
        self._add_instruction(instruction)
        idx = len(self.instructions) - 1
        self.discard(routing_qubits) 
        for inst in self.instructions:
            print(inst)
        return idx

    def discard(self, patches: list[tuple[int, int]], conditioned_on_idx: set[int] = set(), t_gate_bool: bool=False):
        if len(patches) == 0:
            return
        for patch in patches:
            instruction = Instruction('DISCARD', len(self.instructions), frozenset([patch]), 0, conditioned_on_completion_idx=frozenset(conditioned_on_idx), t_gate_bool=t_gate_bool)
            # error out if we've already discarded this patch
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name == 'DISCARD':
                raise ValueError(f'Tried to discard the same patch {patch} twice at instruction {len(self.instructions)}.')
            self._add_instruction(instruction) # add DISCARDS instruction at the end
            for idx in conditioned_on_idx: # if we're conditioned on some inst, update that inst as well to store this new inst since it's conditioned on that inst
                update_instr = self.instructions[idx]
                self.instructions[idx] = Instruction(
                    update_instr.name,
                    update_instr.idx,
                    update_instr.patches,
                    update_instr.duration,
                    update_instr.conditioned_on_idx,
                    update_instr.conditional_dependencies,
                    update_instr.conditioned_on_completion_idx,
                    update_instr.conditional_completion_dependencies  | frozenset([len(self.instructions) - 1]),
                    update_instr.merge_faces,
                    update_instr.group_instr_indices,
                    update_instr.group_name,
                    update_instr.t_gate_bool,
                )

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D, t_gate_bool: bool=False):
        # add an IDLE instruction
        if isinstance(num_rounds, Duration) or num_rounds > 0:
            for patch in patches:
                instruction = Instruction('IDLE', len(self.instructions), frozenset([patch]), num_rounds, t_gate_bool=t_gate_bool) # this IDLE operation should take num_rounds 
                self._add_instruction(instruction)
        elif num_rounds < 0:
            raise ValueError('Number of rounds must be nonnegative.')
        
    # TODO added this new marker object
    def marker(self, patches: list[tuple[int, int]]):
        for patch in patches:
            instruction = Instruction('MARKER', len(self.instructions), frozenset([patch]), 1) # we want the duration to only last 1 round for this marker

    def to_dag(self, d: int | None = None) -> nx.DiGraph:
        """Generate a DAG representations of the instruction indices, with
        'duration' attributes on the nodes. Each edge weight is set to the
        duration of the source instruction of the edge.

        Args:
            d: Temporal code distance. If given, instruction durations will be
                converted from abstract Duration values to actual integer
                durations.

        Returns:
            A directed acyclic graph representing the schedule. Each node
            corresponds to an instruction index and has a 'duration' attribute.
            Each edge corresponds to a dependency between instructions and has
            a 'weight' attribute corresponding to the duration of the source
            instruction.
        """
        if self.generate_dag_incrementally: # a dag is already generated
            dag = self._generated_dag.copy()
            nx.set_edge_attributes(
                G=dag, 
                values={e: self.get_true_duration(self.instructions[e[0]].duration, distance=d) for e in dag.edges()}, # get the duration of the source inst, convert it to actual integer durations if given a temporal code distance, then set that as the edge weight
                name='weight',
            )
            nx.set_node_attributes(
                G=dag,
                values={idx: self.get_true_duration(self.instructions[idx].duration, distance=d) for idx in dag.nodes()}, # for each node, set its attribute as its true duration
                name='duration',
            )
            return dag
        else: # we have to manually create the dag, based on the schedule
            full_schedule = self.full_schedule() # get a list of all instructions
            dag = nx.DiGraph()
            for i,instruction in enumerate(full_schedule.instructions):
                dag.add_node(i, duration=self.get_true_duration(instruction.duration, distance=d)) # add a node for each instruction (specifically its idx), label it with its true duration
                hidden_patches = set() # patches we will no longer draw connections to
                # create edges based on dependencies: each edge corresopnds to a dependency between instructions. 
                # we're seeing for all the insts in the schedule before it, what insts affect the same patches, and if they affect the same patches, that means they are dependent on each other
                # we do it in reverse order because we only need to connect each patch in the new instruction to the MOST RECENT earlier instruction that used the same patch 
                for j,instr in reversed(list(enumerate(full_schedule.instructions[:i]))): 
                    if (set(instruction.patches) & set(instr.patches)) - hidden_patches:
                        dag.add_edge(j, i, weight=self.get_true_duration(instr.duration, distance=d))

                    # if this inst is already connected to another inst (already has dependency created), we don't want to create another dependency on it, so we hide this inst's patches
                    hidden_patches |= set(instr.patches)
            return dag
    
    def total_duration(self, distance: int):
        """Calculate the duration of the longest path in the schedule DAG."""
        dag = self.to_dag(d=distance)
        return nx.dag_longest_path_length(dag)
    
    def get_true_duration(self, duration: Duration | int, distance: int | None = None):
        """Convert abstract Duration values to actual integer durations.

        Args:
            duration: Abstract Duration value or integer duration.
            distance: Temporal code distance. If given, abstract Duration values
                will be converted to actual integer durations.
        
        Returns:
            Converted duration, or input duration if distance is not given.
        """
        if distance is None:
            return duration
        else:
            return Duration.get_true_duration(duration, distance)
    
    # count how many of a specific type of instruction there are 
    def count_instructions(self, name: str):
        return sum(instr.name == name for instr in self.instructions)
    
    # count how many unique patches all the instructions in the schedule covers
    def total_space_footprint(self):
        return len(set(patch for instr in self.instructions for patch in instr.patches))

    # total spacetime volume for instructions (based on distance given). If name is None, then we sum it for all instructions. If name is not None, then we sum it only for the instructions with the specified name
    def total_instruction_volume(self, distance: int, name: str | None = None):
        return sum(instr.spacetime_volume(distance) for instr in self.instructions if name is None or instr.name == name)

    # add an instruction to the current schedule
    def _add_instruction(self, instruction: Instruction):
        idx = len(self.instructions)
        self.instructions.append(instruction)

        # update dag if we need to generate it incrementally
        if self.generate_dag_incrementally:
            self._generated_dag.add_node(idx, duration=instruction.duration)
            for patch in instruction.patches:
                if patch in self._instructions_by_patch and (self._instructions_by_patch[patch][-1], idx) not in self._generated_dag.edges(): # if we have not already created an edge/dependenecy btwn this patch's inst and the current inst
                    prev_instruction = self._instructions_by_patch[patch][-1]
                    assert prev_instruction < idx # check that it is indeed a previous instruction
                    self._generated_dag.add_edge(prev_instruction, idx, weight=self.get_true_duration(self.instructions[prev_instruction].duration)) # set the weight as the previous instruction's (source inst's) duration

        # for every patch in this instruction's patches, add to the instructions_by_patch array (if patch does not exist alr, add key=patch, val=[idx]. if it does alr exist, append to existing patch's val the curr idx)
        for patch in instruction.patches:
            self._instructions_by_patch.setdefault(patch, []).append(idx)

    def _get_final_active_patches(self):
        patches = []
        for patch, instr_idxs in self._instructions_by_patch.items():
            instr = self.instructions[instr_idxs[-1]] # get last instruction for this patch
            if instr.name != 'DISCARD': # if this last inst is not a DISCARD inst, then we append this patch to the array of active patches
                patches.append(patch)
        return patches # return all active patches (patches whose last inst is NOT DISCARD)
