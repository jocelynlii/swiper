from swiper.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
from swiper.device_manager import InstructionTask
from typing import Literal, Any, cast, Iterable, TYPE_CHECKING, Callable
from dataclasses import dataclass
import stim
from tqec import BlockGraph
from tqec.utils.position import Position3D
from tqec import compile_block_graph
from collections import defaultdict
import math

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
        
        self.bg = BlockGraph() # initialize our block graph
        self.bg_stim_circ = stim.Circuit()
        self.bg_total_time = 0
        self.cubes = []

    # Create all the Coordinates of a logical qubit patch at (0,0) (no assigning qubit indexes yet)
    def ls_to_blockgraph(self):
        # From schedule, create the block graph -- we want to add to bg
        g = BlockGraph()
        cubes = []
        pipes = []
        seen_patches = []
        unique_counter = 0 # for labels
        bg_time = 0
        patch_last_idx = {}

        for i, inst in enumerate(self.schedule):
            # get the logical qubit patches that this instruction affects, add to mapping btwn idx:affected_logical_patches
            name = inst.instruction.name
            patches = inst.instruction.patches
            duration = inst.instruction.duration

            if name == "IDLE":
                if len(patches) != 1:
                    raise Exception("IDLE patches is not 1")
                
                patches = list(patches)

                if i-1 >= 0 and self.schedule[i-1].instruction.name == "Y_MEAS":
                    bg_time -= 1
                    y_meas_patch = list(self.schedule[i-1].instruction.patches)[0]
                    for patch in seen_patches:
                        if patch != y_meas_patch:
                            cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", f"{duration}"))
                            pipes.append((patch_last_idx[patch], len(cubes)-1))
                            patch_last_idx[patch] = len(cubes)-1
                else:
                    if patches[0] not in seen_patches:
                        cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "P", f"In_Idle_{patches[0][0]}{patches[0][1]}, {duration}"))
                        seen_patches.append(patches[0])
                        patch_last_idx[patches[0]] = len(cubes)-1
                    else:
                        cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", f"{duration}"))
                        pipes.append((patch_last_idx[patches[0]], len(cubes)-1))
                        patch_last_idx[patches[0]] = len(cubes)-1

                    for patch in seen_patches:
                        if patch == (patches[0][0], patches[0][1]):
                            continue

                        cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", f"{duration}"))
                        pipes.append((patch_last_idx[patch], len(cubes)-1))
                        patch_last_idx[patch] = len(cubes)-1


                bg_time += 1

            elif name == "INJECT_T":
                if len(patches) != 1:
                    raise Exception("INJECT_T patches is not 1")
                
                patches = list(patches)
                
                if patches[0] in seen_patches: # TODO make sure to remove patches from seen_patches when they are discarded
                    raise Exception("Shouldn't be injecting T on a patch that already exists")
                
                cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", f"S_{patches[0][0]}{patches[0][1]}_{unique_counter}, {duration}")) # TODO need to make this label unique -- if inject T on same patch twice, need some other counter to make it unique
                unique_counter += 1
                patch_last_idx[patches[0]] = len(cubes)-1

                for patch in seen_patches:
                    cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", f"{duration}"))
                    pipes.append((patch_last_idx[patch], len(cubes)-1))
                    patch_last_idx[patch] = len(cubes)-1

                seen_patches.append(patches[0])
                bg_time += 1
            
            elif name == "MERGE": # TODO handle with routing qubits merge
                new_patches = []
                for patch in patches:
                    if patch not in seen_patches:
                        seen_patches.append(patch)
                        new_patches.append(patch)
                    
                for patch in seen_patches:
                    cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", f"{duration}")) # TODO: ZXX or ZXZ depends on if patches are neighbors in x or y dir -- x dir = ZXZ, y dir = ZXX
                    if patch not in new_patches:
                        pipes.append((patch_last_idx[patch], len(cubes)-1))
                        patch_last_idx[patch] = len(cubes)-1
                    else:
                        patch_last_idx[patch] = len(cubes)-1

                # TODO: DO SOMETHING WITH PIPES for the 2 patches being merged only!
                pipes.append((len(cubes)-2, len(cubes)-1)) # TODO: hard coded for merges with only 2 qubits involved -- need to generalize/change
                bg_time += 1
            
            elif name == "DISCARD":
                # TODO: NEED TO INDICATE NO PIPES HERE
                for patch in patches:
                    seen_patches.remove(patch)

            elif name == "Y_MEAS":
                if len(patches) != 1:
                    raise Exception("INJECT_T patches is not 1")
                
                patches = list(patches)
                
                if patches[0] not in seen_patches:
                    raise Exception("This patch should have been seen at this point if we are doing Y_MEAS on it")

                cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", f"Y_{patches[0][0]}{patches[0][1]}_{unique_counter}, {duration}"))
                unique_counter += 1
                pipes.append((patch_last_idx[patches[0]], len(cubes)-1))
                patch_last_idx[patches[0]] = len(cubes)-1
                
                # for patch in seen_patches:
                #     cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", ""))
                
                # INDICATE NO PIPES HERE
                # for patch in patches:
                #     seen_patches.remove(patch) # after Y_MEAS, this logical qubit patch is dead

                bg_time += 1

            else:
                raise Exception(f"Unrecognzied lattice surgery instruction: {name}")
            
        for cube in cubes:
            print(cube)
        self.cubes = cubes
        for pipe in pipes:
            print(pipe)
        
        for pos, kind, label in cubes:
            g.add_cube(pos, kind, label)

        for p0, p1 in pipes:
            g.add_pipe(cubes[p0][0], cubes[p1][0])

        self.bg = g
        print(bg_time)
        self.bg_total_time = bg_time

        filled = g.fill_ports_for_minimal_simulation()
        fg = filled[0]
        compiled = compile_block_graph(fg.graph)

        print("=== GHZ-3 Lattice Surgery Compilation (with to_rg optimization) ===")
        for k in [1]: # , 2, 3
            circ = compiled.generate_stim_circuit(k=k)
            d = 2 * k + 1
            print(circ)
            self.bg_stim_circ = circ

        # with open("out4.stim", "w") as f:
        #     f.write(str(self.bg_stim_circ))
        #     f.write("\n")  # optional, nice to end with newline

    import stim

    def stim_circ_conversion(self):
        current = stim.Circuit()
        new_circ = stim.Circuit()
        sections = []

        for inst in self.bg_stim_circ:
            current.append(inst)
            if inst.name == "SHIFT_COORDS":
                sections.append(current)
                current = stim.Circuit()

        # handle leftover (but there shouldn't be any leftover)
        if len(str(current).strip()) > 0:
            sections.append(current)

        print(f"sections {len(sections)}")
        print(f"bg total time {self.bg_total_time}")

        if len(sections) != self.d*self.bg_total_time:
            raise Exception("something went wrong")
        
        cubes_split_time_dict = defaultdict(list)
        for cube in self.cubes:
            pos, kind, label = cube
            cubes_split_time_dict[pos.z].append(cube)

        cubes_split_time_list = [cubes_split_time_dict[z] for z in sorted(cubes_split_time_dict)]
        print(len(cubes_split_time_list))

        sections_by_dist = [sections[i:i+self.d] for i in range(0, len(sections), self.d)]

        if len(cubes_split_time_list) != len(sections_by_dist):
            raise Exception("something went wrong pt 2")
        
        print(cubes_split_time_list)

        # print(sections_by_dist[0:2])
        
        # TODO: this is under the (possibly very bold) assumption that every timestep will have the exact same duration between all idle patches
        for idx, cubes_list in enumerate(cubes_split_time_list):
            curr_section = sections_by_dist[idx]

            pos, kind, label = cubes_list[0]
            duration = label.split(",")[-1].strip()

            # TODO: IMPLEMENT (but doesn't matter for d=3 bc half_d_plus_2 = 3 = d in this specific case)
            if duration != "D":
                if duration == "HALF_D_PLUS_2":
                    # try to modify the middle one
                    int_duration = math.floor(0.5*self.d+2)

                    # if int_duration != self.d:
                        # TODO

            for cube in cubes_list:
                curPos, curKind, curLabel = cube
                cmd = curLabel.split(",")[0].strip().split("_")[0].strip()
                curPatch = (curPos.x, curPos.y)
                modified_section = curr_section[0]

                modified_circuit = stim.Circuit()
                coords = self.bg_stim_circ.get_final_qubit_coordinates()

                if cmd == "S":
                    for inst in modified_section:
                        if inst.name == "R":
                            
                            qubit_targets_obj = inst.targets_copy()
                            qubit_targets = [t.value for t in qubit_targets_obj]
                            midpoint_qubit = qubit_targets[(self.d*self.d)//2]
                            m1, m2 = coords[midpoint_qubit]
                            r_qubits = []
                            rx_qubits = []

                            for qubit in qubit_targets:
                                if (coords[qubit][0] <= m1 and coords[qubit][1] > m2) or (coords[qubit][0] >= m1 and coords[qubit][1] < m2):
                                    rx_qubits.append(qubit)
                                elif (coords[qubit][0] < m1 and coords[qubit][1] <= m2) or (coords[qubit][0] > m1 and coords[qubit][1] >= m2):
                                    r_qubits.append(qubit)
                                elif coords[qubit][0] == m1 and coords[qubit][1] == m2: # precisely the middle qubit, initialize to S state
                                    rx_qubits.append(qubit) # to apply the S-gate, need to first RX (reset to |+>), then apply S gate on this qubit
                                    midpoint_stim_idx = qubit

                            print(r_qubits)
                            print(rx_qubits)
                            modified_circuit.append("R", r_qubits)
                            modified_circuit.append("RX", rx_qubits)
                            modified_circuit.append("TICK")
                            modified_circuit.append("S", midpoint_qubit)
                        else:
                            modified_circuit.append(inst)

                    # replace with new circuit
                    curr_section[0] = modified_circuit
                    sections_by_dist[idx] = curr_section

                    print(modified_circuit)

                # elif cmd == "Y":
                #     for inst in modified_section:
                #         if inst.name == "RX":
                #             qubit_targets_obj = inst.targets_copy()
                #             qubit_targets = [t.value for t in qubit_targets_obj]
                            
                            
                #         else:
                #             modified_circuit.append(inst)

                #     # replace with new circuit
                #     curr_section[0] = modified_circuit
                #     sections_by_dist[idx] = curr_section

                #     print(modified_circuit)

        print(self.bg_stim_circ)
        print(self.schedule)

        # create circuit with new sections by dist
        final_circuit = stim.Circuit()
        for section in sections_by_dist:
            for circ in section:
                final_circuit += circ

        # print(final_circuit)
        with open("dummy.stim", "w") as f:
            f.write(str(final_circuit))
            f.write("\n")  # optional, nice to end with newline
        print(coords) 

        # for section in sections:

    # TODO eventually also have an add noise function

    

def main():
    # create a lattice surgery schedule
    schedule = LatticeSurgerySchedule(generate_dag_incrementally=True)
    prev_injection_flag = False
    schedule.idle([(0,0)], Duration.D)
    # schedule.idle([(0,0)], 3)
    injection_patch = (0,1) if prev_injection_flag else (1,0)
    schedule.inject_T([injection_patch], True)
    prev_injection_flag = not prev_injection_flag
    idx = len(schedule)
    schedule.merge([(0,0), injection_patch], [], t_gate_bool=True)
    schedule.discard([injection_patch], t_gate_bool=True)
    schedule.S((0,0), injection_patch, idx, t_gate_bool=True)
    
    schedule_insts = schedule.instructions
    with open("dummy_sched.txt", "w") as f:
        f.write("\n".join(str(inst) for inst in schedule_insts))

    for i in schedule_insts:
        print(i)
    schedule_inst_tasks = [InstructionTask(i, instr, -1, -1) for i,instr in enumerate(schedule_insts)]
    for task in schedule_inst_tasks:
        print(task)

    # create lattice surgery to stim compiler
    # TODO change the d's to be consistent man
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
    ls.ls_to_blockgraph()
    ls.stim_circ_conversion()
    # MORE IDLE ROUNDS

    # with open("out.stim", "w") as f:
    #     f.write(str(ls.c))
    #     f.write("\n")  # optional, nice to end with newline

    # Next we need to differentiate between which ones are our data qubits and which ones are our ancilla qubits (and which ancillas are X vs Z)
    # This will help us determine what to reset which qubits as (R vs RX)
    # BUILD DICT W KEY=TUPLE QUBIT PATCH, VAL=QUBIT INDEX FOR QUICK INDEX LOOKUPS
    # TODO: I'm just going based off of the MERGE code -- I don't know if it's technically a Z or X surface code patch or whatever


if __name__ == "__main__":
    main()

# def ls_to_stim(schedule: list[Instruction]):
    