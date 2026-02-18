from swiper.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
from swiper.device_manager import InstructionTask
from typing import Literal, Any, cast, Iterable, TYPE_CHECKING, Callable
from dataclasses import dataclass
import stim

@dataclass
class PhysicalQubit: # a syndrome round is a complete cycle of measuring all the stabilizer (syndrome) qubits across the surface code lattice
    """A physical qubit metadata"""
    patch: tuple[int, int] # spatial coords of the patch (logical qubit rgn) this syndrome measurement corresponds to
    qubit_type: Literal["Data", "X_ancilla", "Z_ancilla"]
    stim_idx: int
    bridge_qubit: bool
    initialized: bool = False

    def __repr__(self):
        return f'Physical Qubit({self.patch}, {self.qubit_type}, {self.stim_idx}, {self.bridge_qubit})'

class LatticeSurgeryToStim:
    def __init__(self, schedule: list[InstructionTask] | None, d: int):
        # store schedule and code distance
        self.schedule = schedule
        self.d = d

        # qubit bookkeeping structures
        self.logical_inst_2_logical_patch: dict[int, frozenset[tuple[int, int]]] = {} # key: inst idx, val: logical patch coords
        self.logical_patch_2_physical_qubits: dict[tuple[int, int], list[PhysicalQubit]] = {} # tuple[int, int] # key: logical patch coords, val: all physical qubits' coords corresponding to this logical qubit patch
        # key: logical bridge qubit patch (midpoint btwn 2 logical qubit patches) e.g. logical patches (0,0) and (1,0) will have (0.5,0) here as the "logical patch"
        # val: all physical qubits corresponding to the bridge qubits between these 2 qubit patches
        self.bridge_logical_2_physical_qubits: dict[tuple[int, int], list[PhysicalQubit]] = {} # tuple[int, int]
        self.physical_qubits: list[PhysicalQubit] = [] # tuple[int, int] # idx = idx of the physical qubit in stim, val = physical qubit coords

        # mapping structures for physical qubits to indices in stim
        self.coord_2_idx: dict[tuple[int, int], int] = {} # PhysicalQubit # not sure if want key to be tuple or PhysicalQubit object

        # (0,0) logical qubit patch coordinates -- everything else is a translation of these qubits
        self.qubits_00: list[PhysicalQubit] = [] # tuple[int, int]
        self.logical_2_physical_qubits_00() # initialize these logical qubit patch coordinates

        # need to keep track of temporal rounds
        self.temporal_round: int = 0

        # keep track of meaesurement rounds
        self.measurement_counter: int = 0 # increment for every measurement
        # key = tuple(int, int) = tuple(stim_idx, temporal_round), val = length of measurement index map during this measurement (to get the proper measurement idx, do len(measurement_map)-val)
        self.measurement_idx_map: dict[tuple[int, int], int] = {} 

        # actual circuit
        self.c = stim.Circuit()

        # some measurement bookeeping struct
        self.last_y_meas_rec: dict[str, int] = {}

    # Create all the Coordinates of a logical qubit patch at (0,0) (no assigning qubit indexes yet)
    def logical_2_physical_qubits_00(self):
        # Create all the one-off border qubits
        # start from (2,0) and take the bottom edge of the square, going right
        x = 2
        ancilla_type = "Z_ancilla" # TODO: might not always start with Z_ancilla, depending on the surface code type
        while x < 2*self.d:
            self.qubits_00.append(PhysicalQubit((x,0), ancilla_type, -1, False)) # (x,0)
            x += 4
            if ancilla_type == "Z_ancilla":
                ancilla_type = "X_ancilla"
            else:
                ancilla_type = "Z_ancilla"
        
        # take the right edge of the square, going up
        y = 2
        if x == 2*self.d:
            while y < 2*self.d:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                y += 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"
        else:
            x = 2 * self.d
            y = 4
            while y < 2*self.d:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                y += 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"

        # create qubits along the top edge of the square, going left
        if y == 2 * self.d:
            x = x - 2
            while x > 0:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                x -= 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"
        else:
            y = 2 * self.d
            x = x - 4
            while x > 0:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                x -= 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"

        # create qubits along left edge of the square, going down
        if x == 0:
            y -= 2
            while y > 0:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                y -= 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"
        else:
            x = 0
            y -= 4
            while y > 0:
                self.qubits_00.append(PhysicalQubit((x,y), ancilla_type, -1, False)) # (x,y)
                y -= 4
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"

        # Create the main central patch of qubits
        for i in range(1, 2*self.d):
            if i % 2 == 1:
                for j in range(1, 2*self.d, 2):
                    self.qubits_00.append(PhysicalQubit((i,j), "Data", -1, False)) # (i,j)
            else:
                if i != 2:
                    # if d is even, then every time we move to a new column, we flip the ancilla type
                    # if d is odd, then every time we move to a new column, we keep the same ancilla type
                    if self.d % 2 == 1:
                        # because we flipped after our last execution, we need to flip it back to keep it the same ancilla type as before
                        if ancilla_type == "Z_ancilla":
                            ancilla_type = "X_ancilla"
                        else:
                            ancilla_type = "Z_ancilla"
                else: # start, initialize ancilla type as X_ancilla (with caveat as specified below)
                    # TODO: Because we started with a Z_ancilla at (2,0), we must start with an X_ancilla for the central ancilla qubits. But, we can flip this, depending on the surface code type
                    ancilla_type = "X_ancilla"

                for j in range(2, 2*self.d, 2):
                    self.qubits_00.append(PhysicalQubit((i,j), ancilla_type, -1, False)) # (i,j)
                    if ancilla_type == "Z_ancilla":
                        ancilla_type = "X_ancilla"
                    else:
                        ancilla_type = "Z_ancilla"

    # TODO: DETERMINE WHAT KIND OF ANCILLA BRIDGE QUBITS ARE
    # TODO: X and Z very well may be flipped here, but I'm just going to go off of the merge example given
    def find_edge_qubits(self, q1: tuple[int, int], q2: tuple[int, int]):
        bridge_qubits = []
        physical_q1 = self.logical_patch_2_physical_qubits[q1]
        physical_q2 = self.logical_patch_2_physical_qubits[q2]
        bridge_physical_qubit_objs = []

        # horizontal or vertical neighbors?
        if (q1[0] == q2[0]):
            # vertical neighbor, so want to find qubits with max y for every x
            if q1[1]<q2[1]: # q1 below q2
                min_by_x = {}
                for pq in physical_q2:
                    x, y = pq.patch
                    if x not in min_by_x or y < min_by_x[x][1]:
                        min_by_x[x] = (x, y)

                max_by_x = {}
                for pq in physical_q1:
                    x, y = pq.patch
                    if x not in max_by_x or y > max_by_x[x][1]:
                        max_by_x[x] = (x, y)       

                for x, coord1 in min_by_x.items():
                    coord2 = max_by_x[x]
                    bridge_qubits.append((int((coord1[0]+coord2[0])/2), int((coord1[1]+coord2[1])/2)))

            else: # q2 below q1
                min_by_x = {}
                for pq in physical_q1:
                    x, y = pq.patch
                    if x not in min_by_x or y < min_by_x[x][1]:
                        min_by_x[x] = (x, y)

                max_by_x = {}
                for pq in physical_q2:
                    x, y = pq.patch
                    if x not in max_by_x or y > max_by_x[x][1]:
                        max_by_x[x] = (x, y)       

                for x, coord1 in min_by_x.items():
                    coord2 = max_by_x[x]
                    bridge_qubits.append((int((coord1[0]+coord2[0])/2), int((coord1[1]+coord2[1])/2)))

            # # convert bridge qubits into physical qubits list
            # for bq in bridge_qubits:
            #     bridge_physical_qubit_objs
        else:
            # horizontal neighbor, so want to find qubits with max x for every y
            if q1[0]<q2[0]: # q1 to the left of q2
                min_by_y = {}
                for pq in physical_q2:
                    x, y = pq.patch
                    if y not in min_by_y or x < min_by_y[y][0]:
                        min_by_y[y] = (x, y)

                max_by_y = {}
                for pq in physical_q1:
                    x, y = pq.patch
                    if y not in max_by_y or x > max_by_y[y][0]:
                        max_by_y[y] = (x, y)       

                for y, coord1 in min_by_y.items():
                    coord2 = max_by_y[y]
                    bridge_qubits.append((int((coord1[0]+coord2[0])/2), int((coord1[1]+coord2[1])/2)))

            else: # q2 to the left of q1
                min_by_y = {}
                for pq in physical_q1:
                    x, y = pq.patch
                    if y not in min_by_y or x < min_by_y[y][0]:
                        min_by_y[y] = (x, y)

                max_by_y = {}
                for pq in physical_q2:
                    x, y = pq.patch
                    if y not in max_by_y or x > max_by_y[y][0]:
                        max_by_y[y] = (x, y)

                for y, coord1 in min_by_y.items():
                    coord2 = max_by_y[y]
                    bridge_qubits.append((int((coord1[0]+coord2[0])/2), int((coord1[1]+coord2[1])/2)))

            # convert bridge qubits into physical qubits list
            bridge_qubits.sort(key=lambda x: x[1]) # first sort by y coordinate
            ancilla_type = "Z_ancilla" # TODO: Depending on the type of surface code/boundaries between the 2 patches, this can also be X_ancilla possibly!
            for bq in bridge_qubits:
                bridge_physical_qubit_objs.append(PhysicalQubit(bq, ancilla_type, -1, True))
                if ancilla_type == "Z_ancilla":
                    ancilla_type = "X_ancilla"
                else:
                    ancilla_type = "Z_ancilla"
        

        return bridge_physical_qubit_objs # bridge_qubits
    
    # initialize all the physical qubits of a single logical qubit patch
    def logical_qubit_2_physical_qubits(self, patch: tuple[int, int]):
        # if we haven't seen this logical qubit patch yet, create the physical qubits list for this patch
        physical_qubits = []

        # if the patch is (0,0), we've already created the physical qubits for this patch, in qubits_00
        if patch == (0,0):
            physical_qubits = self.qubits_00
            self.logical_patch_2_physical_qubits[patch] = physical_qubits
            return # physical_qubits

        # if the patch is not (0,0), we know its physical qubits are just a translation of the physical qubits of logical qubit patch (0,0)
        p1, p2 = patch
        # physical_qubits = [(x + p1*(2*(self.d+1)), y + p2*(2*(self.d+1))) for (x, y) in self.qubits_00]
        # carry over the same qubit type
        physical_qubits = [PhysicalQubit((pq.patch[0] + p1*(2*(self.d+1)), pq.patch[1] + p2*(2*(self.d+1))), pq.qubit_type, -1, pq.bridge_qubit) for pq in self.qubits_00]
        self.logical_patch_2_physical_qubits[patch] = physical_qubits
        return # physical_qubits
    
    def add_bridge_qubits(self, patch: tuple[int, int]):
        p1, p2 = patch

        # add bridge qubits
        # first check if we have initialized this current logical patch's neighbor that we want to create a bridge to
        # check all 4 neighbors
        need_bridges_patches = []
        if (p1+1,p2) in self.logical_patch_2_physical_qubits:
            need_bridges_patches.append((p1+1,p2))

        if (p1-1,p2) in self.logical_patch_2_physical_qubits:
            need_bridges_patches.append((p1-1,p2))

        if (p1,p2+1) in self.logical_patch_2_physical_qubits:
            need_bridges_patches.append((p1,p2+1))

        if (p1,p2-1) in self.logical_patch_2_physical_qubits:
            need_bridges_patches.append((p1,p2-1))

        # then check if a bridge already exists btwn the 2 logical patches (we've already initialized it before)
        # still at LOGICAL level here
        create_bridges_patches = []
        for neighbor_patch in need_bridges_patches:
            n1, n2 = neighbor_patch
            if ((n1+p1)/2, (n2+p2)/2) not in self.bridge_logical_2_physical_qubits:
                # create_bridges_patches.append(((n1+p1)/2, (n2+p2)/2))
                create_bridges_patches.append(neighbor_patch)

        final_bridges_patches = []
        # Create the bridge physical qubits between 2 logical patches whose bridge qubits haven't been initialized yet
        for bridge_patch in create_bridges_patches:
            if patch not in self.logical_patch_2_physical_qubits or bridge_patch not in self.logical_patch_2_physical_qubits:
                raise Exception("These qubits don't exist!")
            
            self.bridge_logical_2_physical_qubits[((bridge_patch[0]+patch[0])/2, (bridge_patch[1]+patch[1])/2)] = self.find_edge_qubits(patch, bridge_patch)
            final_bridges_patches.append(((bridge_patch[0]+patch[0])/2, (bridge_patch[1]+patch[1])/2))

        return final_bridges_patches


    # initialize all physical qubits needed for all logical qubits
    def logical_2_physical_qubits(self):
        # take every instruction in the schedule
        for inst in self.schedule:
            # get the logical qubit patches that this instruction affects
            idx = inst.instruction_idx
            logical_patches = inst.instruction.patches
            self.logical_inst_2_logical_patch[idx] = logical_patches

            # for every patch that this instruction touches
            for patch in logical_patches:
                # if we've already seen this patch before, then simply continue (because we've already initialized this patch's physical qubits)
                if patch in self.logical_patch_2_physical_qubits:
                    continue

                # if we haven't seen this logical qubit patch yet, create the physical qubits list for this patch
                physical_qubits = []

                # if the patch is (0,0), we've already created the physical qubits for this patch, in qubits_00
                if patch == (0,0):
                    physical_qubits = self.qubits_00
                    self.logical_patch_2_physical_qubits[patch] = physical_qubits
                    continue

                # if the patch is not (0,0), we know its physical qubits are just a translation of the physical qubits of logical qubit patch (0,0)
                p1, p2 = patch
                # physical_qubits = [(x + p1*(2*(self.d+1)), y + p2*(2*(self.d+1))) for (x, y) in self.qubits_00]
                # carry over the same qubit type
                physical_qubits = [PhysicalQubit((pq.patch[0] + p1*(2*(self.d+1)), pq.patch[1] + p2*(2*(self.d+1))), pq.qubit_type, -1, pq.bridge_qubit) for pq in self.qubits_00]
                self.logical_patch_2_physical_qubits[patch] = physical_qubits

                # add bridge qubits
                # first check if we have initialized this current logical patch's neighbor that we want to create a bridge to
                # check all 4 neighbors
                need_bridges_patches = []
                if (p1+1,p2) in self.logical_patch_2_physical_qubits:
                    need_bridges_patches.append((p1+1,p2))

                if (p1-1,p2) in self.logical_patch_2_physical_qubits:
                    need_bridges_patches.append((p1-1,p2))

                if (p1,p2+1) in self.logical_patch_2_physical_qubits:
                    need_bridges_patches.append((p1,p2+1))

                if (p1,p2-1) in self.logical_patch_2_physical_qubits:
                    need_bridges_patches.append((p1,p2-1))

                # then check if a bridge already exists btwn the 2 logical patches (we've already initialized it before)
                # still at LOGICAL level here
                create_bridges_patches = []
                for neighbor_patch in need_bridges_patches:
                    n1, n2 = neighbor_patch
                    if ((n1+p1)/2, (n2+p2)/2) not in self.bridge_logical_2_physical_qubits:
                        # create_bridges_patches.append(((n1+p1)/2, (n2+p2)/2))
                        create_bridges_patches.append(neighbor_patch)

                # Create the bridge physical qubits between 2 logical patches whose bridge qubits haven't been initialized yet
                for bridge_patch in create_bridges_patches:
                    if patch not in self.logical_patch_2_physical_qubits or bridge_patch not in self.logical_patch_2_physical_qubits:
                        raise Exception("These qubits don't exist!")
                    
                    self.bridge_logical_2_physical_qubits[((bridge_patch[0]+patch[0])/2, (bridge_patch[1]+patch[1])/2)] = self.find_edge_qubits(patch, bridge_patch)
                    

                # for bridge_patch in create_bridges_patches:
                #     bridge_physical_qubits = []

                #     b1, b2 = bridge_patch

                #     lq1 = (0,0)
                #     lq2 = (0,0)
                #     if b1 % 1 == 0.5:
                #         lq1 = (b1-0.5, b2)
                #         lq2 = (b1+0.5, b2)
                #     else:
                #         lq1 = (b1, b2-0.5)
                #         lq2 = (b1, b2+0.5)

                #     if lq1 not in self.logical_patch_2_physical_qubits or lq2 not in self.logical_patch_2_physical_qubits:
                #         raise Exception("These qubits don't exist!")

                    # get the 

                # # assign physical qubit coordinates to the logical patch coord
                # starting_qubit_idx = len(self.physical_qubits)
                # for i in range(0, 2*self.d*self.d-1):
                #     physical_qubits.append()

    def initialize_all_stim_qubits(self):
        # iterate through all logical patch's physical qubits
        for logical_patch_physical_qubits in self.logical_patch_2_physical_qubits.values():
            for physical_qubit in logical_patch_physical_qubits:
                self.c.append('QUBIT_COORDS', len(self.physical_qubits), physical_qubit.patch)
                physical_qubit.stim_idx = len(self.physical_qubits)
                self.coord_2_idx[physical_qubit.patch] = len(self.physical_qubits)
                self.physical_qubits.append(physical_qubit)

        # iterate through all bridge qubits
        for bridge_physical_qubits in self.bridge_logical_2_physical_qubits.values():
            for physical_qubit in bridge_physical_qubits:
                self.c.append('QUBIT_COORDS', len(self.physical_qubits), physical_qubit.patch)
                physical_qubit.stim_idx = len(self.physical_qubits)
                self.coord_2_idx[physical_qubit.patch] = len(self.physical_qubits)
                self.physical_qubits.append(physical_qubit)

        self.c.append("TICK") # append TICK after initialize all qubits

    def initialize_given_stim_qubits(self, logical_patches: list[tuple[int, int]], bridge_logical_patches: list[tuple[int, int]]):
        # iterate through given logical patch's physical qubits
        for lp in logical_patches:
            logical_patch_physical_qubits = self.logical_patch_2_physical_qubits[lp]
            for physical_qubit in logical_patch_physical_qubits:
                self.c.append('QUBIT_COORDS', len(self.physical_qubits), physical_qubit.patch)
                physical_qubit.stim_idx = len(self.physical_qubits)
                self.coord_2_idx[physical_qubit.patch] = len(self.physical_qubits)
                self.physical_qubits.append(physical_qubit)

        # iterate through given bridge qubits
        for blp in bridge_logical_patches:
            bridge_physical_qubits = self.bridge_logical_2_physical_qubits[blp]
            for physical_qubit in bridge_physical_qubits:
                self.c.append('QUBIT_COORDS', len(self.physical_qubits), physical_qubit.patch)
                physical_qubit.stim_idx = len(self.physical_qubits)
                self.coord_2_idx[physical_qubit.patch] = len(self.physical_qubits)
                self.physical_qubits.append(physical_qubit)

        self.c.append("TICK") # append TICK after initialize all qubits

    def restart_all_qubits(self):
        # restart all qubits -- this is for the initial reset for all qubits

        # first initialize all data qubits and reset them using R
        all_data_qubits = [q.stim_idx for q in self.physical_qubits if q.qubit_type == "Data"]
        self.c.append("R", all_data_qubits)

        # append TICK after reset data qubits
        self.c.append("TICK") 

        # then initialize all ancilla qubits
        self.c += self.reset_ancillas()

    def restart_uninitialized_qubits(self):
        # restart uninitialized qubits

        # first initialize all data qubits and reset them using R
        all_data_qubits = []
        for q in self.physical_qubits:
            if (q.qubit_type == "Data" and not q.initialized):
                all_data_qubits.append(q.stim_idx)
                q.initialized = True
        # all_data_qubits = [q.stim_idx for q in self.physical_qubits if (q.qubit_type == "Data" and not q.initialized)]
        self.c.append("R", all_data_qubits)

        # append TICK after reset data qubits
        self.c.append("TICK") 

        # then initialize all ancilla qubits
        # self.c += self.reset_ancillas()

    def reset_ancillas(self): # we always want to reset all ancillas all rounds, so we can leave this as-is
        # create temp circuit that we add these insts to
        circuit = stim.Circuit()

        # first initialize all Z ancillas
        all_Z_ancillas = [q.stim_idx for q in self.physical_qubits if q.qubit_type == "Z_ancilla" and not q.bridge_qubit]
        # self.c.append("R", all_Z_ancillas)
        circuit.append("R", all_Z_ancillas)

        # then initialize all X ancillas
        all_X_ancillas = [q.stim_idx for q in self.physical_qubits if q.qubit_type == "X_ancilla" and not q.bridge_qubit]
        # self.c.append("RX", all_X_ancillas)
        circuit.append("RX", all_X_ancillas)

        # append TICK after reset ancilla qubits
        # self.c.append("TICK")
        circuit.append("TICK")

        return circuit

    def steady_state_CX(self):
        # create temp circuit that we add these insts to
        circuit = stim.Circuit()

        # do the first batch of CX's, for (x+1, y+1) neighbors
        # Pattern: for all data qubits who have an upper right neighbor that's not a bridge qubit, 
        # if the neighbor is an X ancilla then CX ancilla data, else if neighbor is a Z ancill athen CX data ancilla
        up_right_neighbors = []
        for physical_qubits in self.logical_patch_2_physical_qubits.values():
            for pq in physical_qubits:
                # only need to examine for data qubits
                if pq.qubit_type=="Data":
                    neighbor_11 = next((q for q in physical_qubits if q.patch == (pq.patch[0]+1, pq.patch[1]+1)), None)

                    if neighbor_11 is not None:
                        if neighbor_11.qubit_type=="X_ancilla":
                            up_right_neighbors.append(neighbor_11.stim_idx)
                            up_right_neighbors.append(pq.stim_idx)
                            # self.c.append('CX', [neighbor_11.stim_idx, pq.stim_idx])
                        elif neighbor_11.qubit_type=="Z_ancilla":
                            up_right_neighbors.append(pq.stim_idx)
                            up_right_neighbors.append(neighbor_11.stim_idx)
                            # self.c.append('CX', [pq.stim_idx, neighbor_11.stim_idx])

        circuit.append('CX', up_right_neighbors)
        circuit.append("TICK") 

        # Second batch of CX's
        # Pattern: start from either an X ancilla or data qubit, then their upper left neighbor must be either data or Z ancilla (and must exist)
        # connection is always self to neighbor (CX self neighbor)
        # TODO: might need to change the X and Z ancilla type here depending on the surface code type!
        up_left_neighbors = []
        for physical_qubits in self.logical_patch_2_physical_qubits.values():
            for pq in physical_qubits:
                if pq.qubit_type=="Data" or pq.qubit_type=="X_ancilla":
                    neighbor = next((q for q in physical_qubits if q.patch == (pq.patch[0]-1, pq.patch[1]+1)), None)

                    if neighbor is not None:
                        if neighbor.qubit_type=="Z_ancilla" or neighbor.qubit_type=="Data":
                            up_left_neighbors.append(pq.stim_idx)
                            up_left_neighbors.append(neighbor.stim_idx)
                            # self.c.append('CX', [neighbor_11.stim_idx, pq.stim_idx])

        circuit.append('CX', up_left_neighbors)
        circuit.append("TICK") 

        # Third batch of CX's
        # Pattern: start from either an X ancilla or data qubit, then their bottom right neighbor must be either data or Z ancilla (and must exist)
        # connection is always self to neighbor (CX self neighbor)
        # TODO: might need to change the X and Z ancilla type here depending on the surface code type!
        bot_right_neighbors = []
        for physical_qubits in self.logical_patch_2_physical_qubits.values():
            for pq in physical_qubits:
                if pq.qubit_type=="Data" or pq.qubit_type=="X_ancilla":
                    neighbor = next((q for q in physical_qubits if q.patch == (pq.patch[0]+1, pq.patch[1]-1)), None)

                    if neighbor is not None:
                        if neighbor.qubit_type=="Z_ancilla" or neighbor.qubit_type=="Data":
                            bot_right_neighbors.append(pq.stim_idx)
                            bot_right_neighbors.append(neighbor.stim_idx)
                            # self.c.append('CX', [neighbor_11.stim_idx, pq.stim_idx])

        circuit.append('CX', bot_right_neighbors)
        circuit.append("TICK")

        # do the fourth batch of CX's, for (x-1, y-1) neighbors
        # Pattern: for all data qubits who have an bottom left neighbor that's not a bridge qubit, 
        # if the neighbor is an X ancilla then CX ancilla data, else if neighbor is a Z ancill athen CX data ancilla
        bot_left_neighbors = []
        for physical_qubits in self.logical_patch_2_physical_qubits.values():
            for pq in physical_qubits:
                # only need to examine for data qubits
                if pq.qubit_type=="Data":
                    neighbor = next((q for q in physical_qubits if q.patch == (pq.patch[0]-1, pq.patch[1]-1)), None)

                    if neighbor is not None:
                        if neighbor.qubit_type=="X_ancilla":
                            bot_left_neighbors.append(neighbor.stim_idx)
                            bot_left_neighbors.append(pq.stim_idx)
                            # self.c.append('CX', [neighbor.stim_idx, pq.stim_idx])
                        elif neighbor.qubit_type=="Z_ancilla":
                            bot_left_neighbors.append(pq.stim_idx)
                            bot_left_neighbors.append(neighbor.stim_idx)
                            # self.c.append('CX', [pq.stim_idx, neighbor.stim_idx])

        circuit.append('CX', bot_left_neighbors)
        circuit.append("TICK") 

        return circuit

    def measure_and_detectors(self, repeat: bool, repeat_rounds: int = None):
        # create temp circuit that we add these insts to
        circuit = stim.Circuit()

        # if repeat, then we just want to keep our temporal round to be the same (not increment it)
        if repeat:
            curr_temporal_round = self.temporal_round - 1
        else:
            curr_temporal_round = self.temporal_round

        # first measure all ancillas, based on whether they're an X or Z ancilla
        m_qubits = []
        mx_qubits = []
        for physical_qubits in self.logical_patch_2_physical_qubits.values():
            for pq in physical_qubits:
                if pq.qubit_type == "X_ancilla":
                    # mx_qubits.append(pq.stim_idx)
                    mx_qubits.append(pq)
                elif pq.qubit_type == "Z_ancilla":
                    # m_qubits.append(pq.stim_idx)
                    m_qubits.append(pq)
        
        # Do all measurements
        # TODO: fornot first round cases, if we see that we have a repeated measurement of the same patch for more than 3 rounds, then we can get rid of the oldest ones (ensure mem doesn't blow up)
        # self.c.append('M', m_qubits)
        circuit.append('M', [q.stim_idx for q in m_qubits])
        for qubit in m_qubits:
            self.measurement_idx_map[(qubit.stim_idx, self.temporal_round)] = len(self.measurement_idx_map)
            # at any moment in time, I only need to store my current temporal round, and the past 2 temporal rounds' measurements (technically only need past 1, but do past 2 for safety)
            # therefore this means that I can pop the past 3rd measurement out of my map
            val = self.measurement_idx_map.pop((qubit.stim_idx, self.temporal_round-3), None)
            if val is not None:
                for k, v in self.measurement_idx_map.items():
                    self.measurement_idx_map[k] = v - 1

        # self.c.append('MX', mx_qubits)
        circuit.append('MX', [q.stim_idx for q in mx_qubits])
        for qubit in mx_qubits:
            self.measurement_idx_map[(qubit.stim_idx, self.temporal_round)] = len(self.measurement_idx_map)
            val = self.measurement_idx_map.pop((qubit.stim_idx, self.temporal_round-3), None)
            if val is not None:
                for k, v in self.measurement_idx_map.items():
                    self.measurement_idx_map[k] = v - 1
        
        # next, create detectors. This will be different for the first round versus for all other rounds
        print(self.measurement_idx_map)
        if self.temporal_round == 0:
            for q in m_qubits:
                look_back_idx = self.measurement_idx_map[(q.stim_idx, self.temporal_round)]-len(self.measurement_idx_map)
                print(look_back_idx)
                circuit.append("DETECTOR", 
                              [stim.target_rec(look_back_idx)], 
                              [q.patch[0], q.patch[1], curr_temporal_round]) # self.temporal_round
        else:
            for q in m_qubits:
                look_back_idx1 = self.measurement_idx_map[(q.stim_idx, self.temporal_round)]-len(self.measurement_idx_map)
                look_back_idx2 = self.measurement_idx_map[(q.stim_idx, self.temporal_round-1)]-len(self.measurement_idx_map)
                circuit.append("DETECTOR", 
                              [stim.target_rec(look_back_idx1), stim.target_rec(look_back_idx2)], 
                              [q.patch[0], q.patch[1], curr_temporal_round]) # self.temporal_round
                
            for q in mx_qubits:
                look_back_idx1 = self.measurement_idx_map[(q.stim_idx, self.temporal_round)]-len(self.measurement_idx_map)
                look_back_idx2 = self.measurement_idx_map[(q.stim_idx, self.temporal_round-1)]-len(self.measurement_idx_map)
                circuit.append("DETECTOR", 
                              [stim.target_rec(look_back_idx1), stim.target_rec(look_back_idx2)], 
                              [q.patch[0], q.patch[1], curr_temporal_round]) # self.temporal_round

        circuit.append("TICK")

        # TODO: handle bookeeping in the case of repeat -- we want the measurement index maps to still be up-to-date
        # But technically we only need to keep the last 3 most recent rounds of the measurements so we can cheat our bookeeping using this
        if repeat:
            print(repeat_rounds)
            end_round = self.temporal_round + repeat_rounds
            if repeat_rounds >= 3:
                # I'm pretty sure if we have more than 3 repeat rounds and we have our 3 last ones, we can just delete everything currently existing in measurement_idx_map
                self.measurement_idx_map.clear()
                for r in range(end_round-3, end_round):
                    for qubit in m_qubits:
                        self.measurement_idx_map[(qubit.stim_idx, r)] = len(self.measurement_idx_map)
                        # at any moment in time, I only need to store my current temporal round, and the past 2 temporal rounds' measurements (technically only need past 1, but do past 2 for safety)

                    for qubit in mx_qubits:
                        self.measurement_idx_map[(qubit.stim_idx, r)] = len(self.measurement_idx_map)
            elif repeat_rounds > 0: # for fewer than 3 repeat rounds (but greater than 0), need to manually pop everything from measurement index map and only andle the rounds that we do have for updating measurement map
                print(end_round)
                print(repeat_rounds)
                for r in range(end_round-repeat_rounds, end_round):
                    for qubit in m_qubits:
                        if (qubit.stim_idx, r) not in self.measurement_idx_map:
                            self.measurement_idx_map[(qubit.stim_idx, r)] = len(self.measurement_idx_map)
                            # at any moment in time, I only need to store my current temporal round, and the past 2 temporal rounds' measurements (technically only need past 1, but do past 2 for safety)
                            val = self.measurement_idx_map.pop((qubit.stim_idx, r-3), None)
                            if val is not None:
                                for k, v in self.measurement_idx_map.items():
                                    self.measurement_idx_map[k] = v - 1

                    for qubit in mx_qubits:
                        if (qubit.stim_idx, r) not in self.measurement_idx_map:
                            self.measurement_idx_map[(qubit.stim_idx, r)] = len(self.measurement_idx_map)
                            val = self.measurement_idx_map.pop((qubit.stim_idx, r-3), None)
                            if val is not None:
                                for k, v in self.measurement_idx_map.items():
                                    self.measurement_idx_map[k] = v - 1

            self.temporal_round += repeat_rounds
        


        return circuit

    def initialize_qubits(self):
        # initialize the entire physical qubit patches of all logical qubits
        # self.logical_2_physical_qubits()

        # initialize all qubits in stim with their index (QUBIT_COORDS)
        # self.initialize_all_stim_qubits()

        # restart all qubits (data and ancilla) for the first time (R/RX)
        # self.restart_all_qubits()

        # do steady state CX for all qubits
        self.c += self.steady_state_CX()

        # measure all qubits
        self.c += self.measure_and_detectors(repeat=False)

        # done with round 1 - increment temporal round
        self.temporal_round += 1

    def ls_to_stim_fcn(self):
        # first handle initial round
        # self.initialize_qubits()

        # next depending on the instruction, do various things
        for i, inst in enumerate(self.schedule):
            # get the logical qubit patches that this instruction affects, add to mapping btwn idx:affected_logical_patches
            idx = inst.instruction_idx
            logical_patches = inst.instruction.patches
            self.logical_inst_2_logical_patch[idx] = logical_patches

            if inst.instruction.name == "IDLE" and isinstance(inst.instruction.duration, int): # TODO get rid of int check later and implement duration handling (can just use code dist to convert to int)
                # First, we check if any of the logical patches this IDLE touches are patches we have not initialized
                have_new_patch = False
                # Check every single patch this instruction touches
                for patch in inst.instruction.patches:
                    if patch in self.logical_patch_2_physical_qubits: # if this patch is already initialized, we continue
                        continue

                    # if patch is not in our logical patch two physical qubits mapping, we need to initialize its physical qubits because we know it's uninitialized
                    self.logical_qubit_2_physical_qubits(patch) # create the physical qubits for this specific logical qubit patch, update bookkeeping struct
                    bridge_logical_patches = self.add_bridge_qubits(patch) # if adding this logical patch creates 2 adjacent logical patches, then we need to create the bridge qubits btwn them

                    # add the new physical qubits that we initialized to stim - pass in the logical patches/logical bridge patches that correspond to these new physical qubits into the fcn
                    self.initialize_given_stim_qubits([patch], bridge_logical_patches) 

                    self.restart_uninitialized_qubits() # restart all uninitialized physical DATA qubits only

                    # self.c += self.steady_state_CX()
                    # self.c += self.measure_and_detectors(repeat=False)

                    have_new_patch = True # need to keep track of if we have created a new patch -- affects num repeat rounds of IDLE


                # if we initialize a new patch this round, then we only want to REPEAT IDLE for num_rounds-1; else, repeat for num_rounds/duration
                if have_new_patch: # i == 0 # if we start with IDLE as the first instruction 
                    # for the new patch, finish resetting all ancillas (including existing ones), doing CXs (incl. existing ones), and measure/detect (incl. existing ones)
                    self.c += self.reset_ancillas()
                    self.c += self.steady_state_CX()
                    self.c += self.measure_and_detectors(repeat=False) # TODO: here the Z ancillas are first measured, but pretty sure this depends on the type of surface code
                    self.temporal_round += 1 # implement temporal round by 1 because here we just finished our first round

                    # now begin to implement the REPEAT rounds
                    num_idle_rounds = inst.instruction.duration # TODO: need to handle case where this is not an int and instead is some Duration object
                    body = stim.Circuit()
                    body += self.reset_ancillas()
                    body += self.steady_state_CX()
                    body.append("SHIFT_COORDS", [], [0, 0, 1])
                    body += self.measure_and_detectors(repeat=True, repeat_rounds=num_idle_rounds-1)
                    self.c.append(stim.CircuitRepeatBlock(repeat_count=num_idle_rounds-1, body=body))
                    self.c.append("SHIFT_COORDS", [], [0, 0, -1*(num_idle_rounds-1)]) # need to reset our shift here!
                else:
                    num_idle_rounds = inst.instruction.duration # TODO: need to handle case where this is not an int and instead is some Duration object
                    body = stim.Circuit()
                    body += self.reset_ancillas()
                    body += self.steady_state_CX()
                    body.append("SHIFT_COORDS", [], [0, 0, 1])
                    body += self.measure_and_detectors(repeat=True, repeat_rounds=num_idle_rounds)
                    self.c.append(stim.CircuitRepeatBlock(repeat_count=num_idle_rounds, body=body))
                    self.c.append("SHIFT_COORDS", [], [0, 0, -1*(num_idle_rounds)]) # need to reset our shift here!
            elif inst.instruction_name == "INJECT_T":
                # instead of injecting a T state on a logical qubit patch, inject an S state due to stim requiring clifford gates
                # magic state injection based on https://dl.acm.org/doi/pdf/10.1145/3528416.3530237

                # first check if this is a new patch, INJECT_T should always act on a completely new patch (not existing/active patch -- means it was discarded or never used)
                for patch in inst.instruction.patches:
                    if patch in self.logical_patch_2_physical_qubits:
                        raise Exception("This patch should be discarded at this point/never initialized before!")
                    
                    # now initialize this patch's physical qubits (data and ancilla), as according to the MR approach
                    # first initialize this patch's physical qubits
                    self.logical_qubit_2_physical_qubits(patch) # create the physical qubits for this specific logical qubit patch, update bookkeeping struct
                    bridge_logical_patches = self.add_bridge_qubits(patch) # if adding this logical patch creates 2 adjacent logical patches, then we need to create the bridge qubits btwn them

                    # add the new physical qubits that we initialized to stim - pass in the logical patches/logical bridge patches that correspond to these new physical qubits into the fcn
                    self.initialize_given_stim_qubits([patch], bridge_logical_patches) 

                    # now we need to reset these qubits, but specifically in terms of the MR approach
                    # first initialize all data qubits according to the MR approach

                    # then ancillas are reset as normal (in the MR approach, doesn't change how ancillas are prepared)


def main():
    # create a lattice surgery schedule
    schedule = LatticeSurgerySchedule(generate_dag_incrementally=True)
    prev_injection_flag = False
    schedule.idle([(0,0)], 3)
    # schedule.idle([(0,0)], 3)
    injection_patch = (0,1) if prev_injection_flag else (1,0)
    schedule.inject_T([injection_patch], True)
    prev_injection_flag = not prev_injection_flag
    idx = len(schedule)
    schedule.merge([(0,0), injection_patch], [], t_gate_bool=True)
    schedule.discard([injection_patch], t_gate_bool=True)
    schedule.S((0,0), injection_patch, idx, t_gate_bool=True)
    schedule_insts = schedule.instructions
    schedule_inst_tasks = [InstructionTask(i, instr, -1, -1) for i,instr in enumerate(schedule_insts)]
    # for task in schedule_inst_tasks:
    #     print(task)

    # create lattice surgery to stim compiler
    ls = LatticeSurgeryToStim(schedule_inst_tasks, 3)
    # ls.logical_2_physical_qubits_00()

    # Creates physical qubits for all of the schedule's logical qubit patches
    # ls.logical_2_physical_qubits() 
    
    # # Now we basically need to create the actual stim circuit from the schedule
    # # First create all QUBIT_COORDS
    # # FIRST IDLE ROUND
    # ls.initialize_all_stim_qubits()
    # ls.restart_all_qubits()
    # ls.steady_state_CX()
    # ls.measure_and_detectors()
    # ls.temporal_round += 1 # done with round 1!\
    ls.ls_to_stim_fcn()
    # print(ls.physical_qubits)
    # print(ls.bridge_logical_2_physical_qubits)
    # print(ls.logical_patch_2_physical_qubits)
    print(ls.c)
    print(ls.measurement_idx_map)
    print(ls.temporal_round)

    # MORE IDLE ROUNDS

    with open("out.stim", "w") as f:
        f.write(str(ls.c))
        f.write("\n")  # optional, nice to end with newline

    # Next we need to differentiate between which ones are our data qubits and which ones are our ancilla qubits (and which ancillas are X vs Z)
    # This will help us determine what to reset which qubits as (R vs RX)
    # BUILD DICT W KEY=TUPLE QUBIT PATCH, VAL=QUBIT INDEX FOR QUICK INDEX LOOKUPS
    # TODO: I'm just going based off of the MERGE code -- I don't know if it's technically a Z or X surface code patch or whatever


if __name__ == "__main__":
    main()

# def ls_to_stim(schedule: list[Instruction]):
    