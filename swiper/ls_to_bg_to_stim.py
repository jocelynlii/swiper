from swiper.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
from swiper.device_manager import InstructionTask
from typing import Literal, Any, cast, Iterable, TYPE_CHECKING, Callable
from dataclasses import dataclass
import stim
from tqec import BlockGraph
from tqec.utils.position import Position3D
from tqec import compile_block_graph

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
                            cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", ""))
                            pipes.append((patch_last_idx[patch], len(cubes)-1))
                            patch_last_idx[patch] = len(cubes)-1
                else:
                    if patches[0] not in seen_patches:
                        cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "P", f"In_Idle_{patches[0][0]}{patches[0][1]}"))
                        seen_patches.append(patches[0])
                        patch_last_idx[patches[0]] = len(cubes)-1
                    else:
                        cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", ""))
                        pipes.append((patch_last_idx[patches[0]], len(cubes)-1))
                        patch_last_idx[patches[0]] = len(cubes)-1

                    for patch in seen_patches:
                        if patch == (patches[0][0], patches[0][1]):
                            continue

                        cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", ""))
                        pipes.append((patch_last_idx[patch], len(cubes)-1))
                        patch_last_idx[patch] = len(cubes)-1


                bg_time += 1

            elif name == "INJECT_T":
                if len(patches) != 1:
                    raise Exception("INJECT_T patches is not 1")
                
                patches = list(patches)
                
                if patches[0] in seen_patches: # TODO make sure to remove patches from seen_patches when they are discarded
                    raise Exception("Shouldn't be injecting T on a patch that already exists")
                
                cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", f"S_{patches[0][0]}{patches[0][1]}_{unique_counter}")) # TODO need to make this label unique -- if inject T on same patch twice, need some other counter to make it unique
                unique_counter += 1
                patch_last_idx[patches[0]] = len(cubes)-1

                for patch in seen_patches:
                    cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", ""))
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
                    cubes.append((Position3D(patch[0], patch[1], bg_time), "ZXZ", ""))
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

                cubes.append((Position3D(patches[0][0], patches[0][1], bg_time), "ZXZ", f"Y_{patches[0][0]}{patches[0][1]}_{unique_counter}"))
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
        for pipe in pipes:
            print(pipe)
        
        for pos, kind, label in cubes:
            g.add_cube(pos, kind, label)

        for p0, p1 in pipes:
            g.add_pipe(cubes[p0][0], cubes[p1][0])

        filled = g.fill_ports_for_minimal_simulation()
        fg = filled[0]
        compiled = compile_block_graph(fg.graph)

        print("=== GHZ-3 Lattice Surgery Compilation (with to_rg optimization) ===")
        for k in [1]: # , 2, 3
            circ = compiled.generate_stim_circuit(k=k)
            d = 2 * k + 1
            print(circ)

    

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
    for i in schedule_insts:
        print(i)
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
    ls.ls_to_blockgraph()
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
    