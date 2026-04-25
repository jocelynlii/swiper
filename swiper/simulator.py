import sys
from typing import Callable, Any
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime as dt
import inspect
from dataclasses import dataclass, asdict
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.device_manager import DeviceData, DeviceManager
from swiper.decoder_manager import DecoderData, DecoderManager
from swiper.window_manager import WindowData, SlidingWindowManager, ParallelWindowManager, TAlignedWindowManager
from swiper.window_builder import WindowBuilder
import swiper.plot as plotter
import networkx as nx
import json
from pathlib import Path
import csv
import os

# types for all of the simulator's parameters (detailed explanations for each parameter below)
@dataclass
class SimulatorParams:
    distance: int
    scheduling_method: str
    decoding_latency_fn: str | None
    speculation_mode: str | None
    speculation_latency: int
    speculation_accuracy: float
    poison_policy: str
    missed_speculation_modifier: float
    max_parallel_processes: int | None
    lightweight_setting: int
    rng: int | None
    pending_window_count_cutoff: int | None = None
    device_rounds_cutoff: int | None = None
    clock_timeout_seconds: int | None = None
    processor_prediction_results: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)

class DecodingSimulator:
    def __init__(
            self,
        ):
        self._device_manager: DeviceManager | None = None
        self._window_manager: SlidingWindowManager | ParallelWindowManager | TAlignedWindowManager | None = None
        self._decoding_manager: DecoderManager | None = None

    # run function with all of configuration's parameters. Gives default values for some of the parameters
    # returns 
    def run(
            self,
            schedule: LatticeSurgerySchedule,
            distance: int,
            scheduling_method: str,
            decoding_latency_fn: Callable[[int], int],
            speculation_mode: str | None = None,
            speculation_latency: int = 1,
            speculation_accuracy: float = 0,
            poison_policy: str = 'successors',
            missed_speculation_modifier: float = 1.4,
            max_parallel_processes: int | str | None = None,
            progress_bar: bool = False,
            print_interval: dt.timedelta | None = None,
            pending_window_count_cutoff: int = 0,
            device_rounds_cutoff: int = 0,
            clock_timeout: dt.timedelta | None = None,
            save_animation_frames: bool = False,
            lightweight_setting: int = 0,
            rng: int | np.random.Generator = np.random.default_rng(),
            need_full_window: bool = None,
            decoder_parameters: str = None,
            # full_window_dag: nx.DiGraph = None
            window_parameters: str | dict | None = None,
            filename: str = "dummy",
            slow_gate: bool = False,
            slow_gate_mult: int = None,
            spec_strat: str = "default",
            max_depth: int = None,
            spec_depth_threshold: int = None,
        ) -> tuple[bool, SimulatorParams, DeviceData, WindowData, DecoderData]:
        """TODO
        
        Args:
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
            distance: The distance of the surface code (which also specifies the
                number of QEC rounds for each lattice surgery operation).
            scheduling_method: 'sliding', 'parallel', or 'aligned'. See
                WindowManager.
            decoding_latency_fn: A function that returns a decoding latency
                given the spacetime volume of the decoding problem. See
                DecoderManager.
            speculation_mode: 'integrated', 'separate', or None. See
                DecoderManager.
            speculation_latency: The latency of a speculative prediction, in
                units of rounds of QEC. See DecoderManager.
            speculation_accuracy: The probability that a speculative prediction
                is correct. See DecoderManager.
            poison_policy: 'successors' or 'descendants'. See DecoderManager.
            missed_speculation_modifier: See DecoderManager.
            max_parallel_processes: Maximum number of parallel decoding
                processes to run. If None, run as many as possible. If
                'predict', first run an ideal simulation (perfect speculation)
                of the schedule to estimate the maximum number of parallel
                tasks.
            progress_bar: If True, display a progress bar for the simulation.
            pending_window_count_cutoff: If the number of pending windows
                exceeds this value, the simulation is considered to have failed
                and will return early.
            device_rounds_cutoff: If the number of device rounds exceeds this
                value, the simulation is considered to have failed and will
                return early.
            clock_timeout: If given, stop simulation after this much time has
                elapsed.
            save_animation_frames: If using in Jupyter notebook, use %%capture
                TODO: broken
            lightweight_setting: Optimization level for memory usage. Affects
                runtime memory usage and output data size. Some output data will
                not be available at higher settings.
                0: No optimization.
                1: Avoid data structures that scale with the total number of
                    device rounds, but keep some data structures that scale with
                    the number of windows.
                2: Avoid any data structures that scales with simulation
                    duration.
            rng: Random number generator.
        """
        start_time = dt.datetime.now()

        if print_interval is not None: # if printing is not turned off, we print updates
            print(f'{start_time.strftime("%Y-%m-%d %H:%M:%S")} | Starting simulation')
            sys.stdout.flush() # print directly to terminal

        processor_prediction_results = {}
        if max_parallel_processes == 'predict': # predict the maximum number of procssors that we need for running this program
            print('PREDICTION STEP BEGIN---------------------------------')
            if pending_window_count_cutoff > 0 or device_rounds_cutoff > 0:
                raise ValueError("Cannot predict max parallel processes with cutoffs")
            prediction_simulator = DecodingSimulator()
            # run with exact same parameters, except speculation accuracy is 1.0, there's no maximum parallel process, and clock_timeout is cut in half (bc time shld be shorter if don't need to deal with mispredictions)
            # prediction_success, _, pred_device_data, _, pred_decode_data, _, _ = prediction_simulator.run(
            prediction_data, _, _ = prediction_simulator.run(
                schedule=schedule,
                distance=distance,
                scheduling_method=scheduling_method,
                decoding_latency_fn=decoding_latency_fn,
                speculation_mode=speculation_mode,
                speculation_latency=speculation_latency,
                speculation_accuracy=1.0,
                poison_policy=poison_policy,
                missed_speculation_modifier=missed_speculation_modifier,
                max_parallel_processes=None,
                clock_timeout=(clock_timeout/2 if clock_timeout is not None else None),
                lightweight_setting=2,
                rng=rng,
                need_full_window=False, # TODO: ARE U SURE THIS SHLD BE FALSE? I THINK SO BUT NOT SURE
                decoder_parameters=decoder_parameters
            )
            prediction_success, _, pred_device_data, _, pred_decode_data  = prediction_data # , _, _
            assert prediction_success
            # get the max number of proceses from executing the equation to get this number, as defined in the paper
            max_parallel_processes = pred_decode_data.max_parallel_decoders + math.ceil((pred_decode_data.decode_process_volume / pred_decode_data.num_rounds) * (1 - speculation_accuracy))
            # store various metadata from this prediction step
            processor_prediction_results['device_data:num_rounds'] = pred_device_data.num_rounds
            processor_prediction_results['device_data:total_volume'] = pred_device_data.total_volume
            processor_prediction_results['device_data:avg_conditioned_decode_wait_time'] = pred_device_data.avg_conditioned_decode_wait_time
            processor_prediction_results['decode_data:num_rounds'] = pred_decode_data.num_rounds
            processor_prediction_results['decode_data:max_parallel_processes'] = pred_decode_data.max_parallel_decoders
            processor_prediction_results['decode_data:parallel_process_volume'] = pred_decode_data.decode_process_volume
            print(f'Predicted max parallel processes: {max_parallel_processes}')
            print('\nPREDICTION STEP END-----------------------------------\n')
        assert max_parallel_processes is None or isinstance(max_parallel_processes, int)


        # code to get the full window dag by running the simulation first and 


        # self.simulation_params = SimulatorParams(
        #     distance=distance,
        #     scheduling_method=scheduling_method,
        #     decoding_latency_fn=decoding_latency_fn_str,
        #     speculation_mode=speculation_mode,
        #     speculation_latency=speculation_latency,
        #     speculation_accuracy=speculation_accuracy,
        #     poison_policy=poison_policy,
        #     missed_speculation_modifier=missed_speculation_modifier,
        #     max_parallel_processes=max_parallel_processes,
        #     pending_window_count_cutoff=pending_window_count_cutoff,
        #     device_rounds_cutoff=device_rounds_cutoff,
        #     clock_timeout_seconds=(clock_timeout.total_seconds() if clock_timeout else None),
        #     lightweight_setting=lightweight_setting,
        #     rng=(rng if isinstance(rng, int) else None),
        #     processor_prediction_results=processor_prediction_results,
        # )

        # might want to do this with an if-statement, but for now run it always
        # print('FULL WINDOW DAG STEP BEGIN---------------------------------')
        full_window_arr = []
        full_window_dag = None # nx.DiGraph()
        if need_full_window:
            window_dag_simulator = DecodingSimulator()
            # run with exact same parameters, except speculation accuracy is 1.0, there's no maximum parallel process, and clock_timeout is cut in half (bc time shld be shorter if don't need to deal with mispredictions)
            _, full_window_arr, full_window_dag = window_dag_simulator.run(
                schedule=schedule,
                distance=distance,
                scheduling_method=scheduling_method,
                decoding_latency_fn=decoding_latency_fn,
                speculation_mode=speculation_mode,
                speculation_latency=speculation_latency,
                speculation_accuracy=speculation_accuracy,
                poison_policy=poison_policy,
                missed_speculation_modifier=missed_speculation_modifier,
                max_parallel_processes=max_parallel_processes,
                clock_timeout=(clock_timeout/2 if clock_timeout is not None else None),
                lightweight_setting=2,
                rng=rng,
                need_full_window=False,
                decoder_parameters=decoder_parameters,
                window_parameters=window_parameters
            )
            # print('\nFULL WINDOW DAG STEP END-----------------------------------\n')
        # assert full_window_arr is not None and full_window_dag is not None
        # print("full window arr ", full_window_arr)
        # print("full window dag ", full_window_dag)

        # initialize actual experiment with these settings
        self.initialize_experiment(
            schedule=schedule,
            distance=distance,
            scheduling_method=scheduling_method,
            decoding_latency_fn=decoding_latency_fn,
            speculation_mode=speculation_mode,
            speculation_latency=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            poison_policy=poison_policy,
            missed_speculation_modifier=missed_speculation_modifier,
            max_parallel_processes=max_parallel_processes,
            lightweight_setting=lightweight_setting,
            rng=rng,
            pending_window_count_cutoff=pending_window_count_cutoff,
            device_rounds_cutoff=device_rounds_cutoff,
            clock_timeout_seconds=(clock_timeout.total_seconds() if clock_timeout else None),
            processor_prediction_results=processor_prediction_results,
            full_window_dag=full_window_dag,
            decoder_parameters=decoder_parameters,
            window_parameters=window_parameters,
            slow_gate=slow_gate,
            slow_gate_mult=slow_gate_mult,
            spec_strat=spec_strat,
            max_depth=max_depth,
            spec_depth_threshold=spec_depth_threshold,
        )
        # make sure we have all the required managers
        assert self._device_manager is not None
        assert self._window_manager is not None
        assert self._decoding_manager is not None

        # bookeeping for if we want additional bookeeping like progress bar and svaing animation frames
        if progress_bar:
            pbar_r = tqdm.tqdm(desc='Surface code rounds')
            # pbar_i = tqdm.tqdm(total=len(schedule.all_instructions), desc='Scheduled instructions complete')

        if save_animation_frames:
            fig = plt.figure()
            self.frame_data = []

        max_window_arr_len = 0
        # print("simulator device manager schedule insts run", self._device_manager.schedule_instructions)
        while not self.is_done():
            self.step_experiment(pending_window_count_cutoff=pending_window_count_cutoff, device_rounds_cutoff=device_rounds_cutoff, print_interval=print_interval)

            # if len(self._window_manager.all_windows)
            # print("all windows", self._window_manager.all_windows)
            # update bookeeping for each step
            if progress_bar and self._decoding_manager._current_round % 100 == 0:
                pbar_r.update(100)
                # pbar_i.update(len(fully_decoded_instructions) - pbar_i.n)
                # pbar_i.refresh()
            if save_animation_frames:
                ax = plotter.plot_device_schedule_trace(self._device_manager.get_data(), spacing=1, default_fig=fig)
                ax.set_zticks([])
                self.frame_data.append(ax)
            # if timeout, we failed
            if clock_timeout is not None and dt.datetime.now() > start_time + clock_timeout:
                self.failed = True
        
        # # # print window info
        # print("all windows ", len(self._window_manager.all_windows)) # , " ", self._window_manager.all_windows
        # print("Nodes:")
        # for n, attrs in self._window_manager.window_dag.nodes(data=True):
        #     print(f"  {n}: {attrs}")

        # print("\nEdges:")
        # for u, v, attrs in self._window_manager.window_dag.edges(data=True):
        #     print(f"  {u} -> {v}: {attrs}")

        # # # print instr info
        # print("all insts ", len(self._device_manager.schedule_dag)) # , " ", self._device_manager.schedule_dag
        # # print("all insts schedule", len(self._device_manager.schedule_instructions), " ", self._device_manager.schedule_instructions)
        # print("Nodes:")
        # for n, attrs in self._device_manager.schedule_dag.nodes(data=True):
        #     print(f"  {n}: {attrs}")

        # print("\nEdges:")
        # for u, v, attrs in self._device_manager.schedule_dag.edges(data=True):
        #     print(f"  {u} -> {v}: {attrs}")


        # og print stuff
        if print_interval is not None:
            print(f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Finished simulation')
            sys.stdout.flush()
            
        if progress_bar:
            pbar_r.update(self._decoding_manager._current_round - pbar_r.n)
            # pbar_i.update(pbar_i.total - pbar_i.n)
            pbar_r.close()
            # pbar_i.close()

        _, _, device_data, _, decoder_data = self.get_data()
        
        
        count_t_windows = 0
        unwanted_idle_windows = 0
        # if scheduling_method != "parallel":
            
        for idx, val in enumerate(self._window_manager.all_windows):
            # print("simulator idx, val", idx, val)
            if val is not None:
                for inst in val.parent_instr_idx:
                    if len(val.parent_instr_idx) == 1:
                        if self._device_manager.schedule_instructions[inst].instruction.name == "IDLE" and not self._device_manager.schedule_instructions[inst].instruction.t_gate_bool:
                            count_t_windows = count_t_windows + 1

                    if inst == -1:
                        unwanted_idle_windows = unwanted_idle_windows + 1
                        break

            # if len(val.parent_instr_idx) == 1:
            #     for inst in val.parent_instr_idx:
            #         if self._device_manager.schedule_instructions[inst].instruction.name == "IDLE" and not self._device_manager.schedule_instructions[inst].instruction.t_gate_bool:
            #             count_t_windows = count_t_windows + 1

        # print("decoder data end ", decoder_data.num_failed_speculations) # , " ", decoder_data.num_discarded_decodes
        # print("total wasted decode volume ", decoder_data.wasted_decode_volume)
        # # if decoder_data.num_rounds != device_data.num_rounds:
        # #     print("WRONG ROUNDS WRONG ROUNDS")
        # print("total number of rounds ", decoder_data.num_rounds)
        # print("total number of rounds device data ", device_data.num_rounds)
        # # print("window decoding times start ", decoder_data.window_decoding_start_times)
        # # print("window decoding times end ", decoder_data.window_decoding_completion_times)
        # # print("per window wasted rounds ", decoder_data.per_window_wasted_rounds)
        # # print("per window poisoned ", decoder_data.per_window_poisoned)
        # # print("per window parent insts ", decoder_data.per_window_parent_inst)
        # print("conditional wait times", device_data.conditioned_decode_wait_times)
        # print("avg conditional wait times", device_data.avg_conditioned_decode_wait_time)
        # print("num t gates", device_data.num_t_gates)
        #     # print("non t windows", count_t_windows)
        # print("unwanted idle windows", unwanted_idle_windows)
        # print("rounds with only unwanted idles", self._decoding_manager._unwanted_idle_rounds)
        # print("total decoder volume", self._decoding_manager._decode_processor_spacetime_volume)
        # print("total num mispredictions", self._decoding_manager._num_failed_speculations)

       

        if need_full_window: # only need to write to file after the second run (not the first run that generates the full window dag)
            row = {
                "decoder data end": decoder_data.num_failed_speculations,
                "total wasted decode volume": decoder_data.wasted_decode_volume,
                "total number of rounds": decoder_data.num_rounds,
                "total number of rounds device data": device_data.num_rounds,
                "per inst windows": decoder_data.per_inst_windows,
                "per window speculation acc": decoder_data.per_window_spec_acc,
                "conditional wait times": str(device_data.conditioned_decode_wait_times),
                "avg conditional wait times": device_data.avg_conditioned_decode_wait_time,
                "unwanted idle windows": unwanted_idle_windows,
                "rounds with only unwanted idles": self._decoding_manager._unwanted_idle_rounds,
                "rounds with any number of unwanted idles": self._decoding_manager._unwanted_idle_rounds_any,
                "unwanted idles volume": self._decoding_manager._unwanted_idle_rounds_volume,
                "total decoder volume": self._decoding_manager._decode_processor_spacetime_volume,
                "total num mispredictions": self._decoding_manager._num_failed_speculations,
                "total num successful predictions": self._decoding_manager._num_successful_speculations,
                "total num speculations": self._decoding_manager._num_successful_speculations+self._decoding_manager._num_failed_speculations,
                "average speculation depth": decoder_data.average_speculation_depth,
                "total num windows": decoder_data.num_completed_windows,
                "avg conditional wait time (not volume)": device_data.avg_conditioned_decode_wait_time_individual,
                "spec depth calc time": self._decoding_manager.total_spec_depth_calc_time,
                "total backlog": decoder_data.total_backlog,
                "average backlog": decoder_data.average_backlog,
                "processor idle rounds": decoder_data.processor_idle_rounds,
                "processor idle percent": decoder_data.processor_idle_percent,
                "average active windows": decoder_data.average_active_windows,
            }

            # filename = f"results_superconducting_{max_parallel_processes}_0.8_new.csv" # _{max_parallel_processes}
            # filename = f"results_superconducting_{max_parallel_processes}_{speculation_accuracy}_specaccvary.csv" # _{max_parallel_processes}
            # filename = f"results_slow_{max_parallel_processes}_{speculation_accuracy}_more.csv" # _{max_parallel_processes}
            # filename = f"results_fast_{max_parallel_processes}_{speculation_accuracy}_shallow_more_ui2.csv"
            # filename = f"results_slow_{max_parallel_processes}_{speculation_accuracy}_shallow_more_ui2.csv"

            # filename = f"results_sequential_fast_{max_parallel_processes}_{speculation_accuracy}_shallow.csv"
            # filename = f"results_sequential_slow_{max_parallel_processes}_{speculation_accuracy}_shallow.csv"
            # filename = f"results_sequential_fast_{max_parallel_processes}_{speculation_accuracy}_shallow.csv"
            # filename = f"results_sequential_slow_{max_parallel_processes}_{speculation_accuracy}_deep.csv"
            # filename = f"results_dummy.csv"
            file_exists = os.path.isfile(filename)

            with open(filename, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)


            


            window_parameters_file = {}
            window_metadata_file_ro = []
            decoding_latency_fn_str = ""
            if scheduling_method != "parallel" and scheduling_method != "aligned":
                for idx, val in enumerate(self._window_manager.all_windows):
                    # try:
                    #     decoding_latency_fn_str = inspect.getsource(val.decoding_time_fn)
                    # except Exception as e:
                    #     print(f'Failed to get source of decoding_latency_fn: {e}')
                    #     decoding_latency_fn_str = None

                    window_key = {}
                    cmt_rgn_list = []
                    buffer_rgn_list = []
                    
                    # construct syndrome round dicts
                    for cmt_rgn in val.commit_region:
                        cmt_rgn_dict = {}
                        cmt_rgn_dict['patch'] = list(cmt_rgn.patch) # need convert tuple to list for json
                        cmt_rgn_dict['duration'] = cmt_rgn.duration
                        cmt_rgn_dict['num_spatial_boundaries'] = cmt_rgn.num_spatial_boundaries
                        cmt_rgn_dict['initialized_patch'] = cmt_rgn.initialized_patch
                        cmt_rgn_dict['discard_after'] = cmt_rgn.discard_after
                        cmt_rgn_list.append(cmt_rgn_dict)

                    for buffer_rgn in val.buffer_regions:
                        buffer_rgn_dict = {}
                        buffer_rgn_dict['patch'] = list(buffer_rgn.patch) # need convert tuple to list for json
                        buffer_rgn_dict['duration'] = buffer_rgn.duration
                        buffer_rgn_dict['num_spatial_boundaries'] = buffer_rgn.num_spatial_boundaries
                        buffer_rgn_dict['initialized_patch'] = buffer_rgn.initialized_patch
                        buffer_rgn_dict['discard_after'] = buffer_rgn.discard_after
                        buffer_rgn_list.append(buffer_rgn_dict)

                    parent_inst_list = list(val.parent_instr_idx)

                    window_key['commit_region'] = cmt_rgn_list
                    window_key['buffer_regions'] = buffer_rgn_list
                    window_key['parent_instr_idx'] = parent_inst_list
                    window_key['constructed'] = val.constructed
                
                    if decoding_latency_fn_str is not None:
                        window_parameters_file[idx] = {
                            'window_info': window_key,
                            'edit_parameters': {
                                'speculation_accuracy': val.speculation_accuracy,
                                'speculation_time': val.speculation_time,
                                'decoding_time_fn': val.decoding_time_fn_str # decoding_latency_fn_str.split("=")[1][1:-1]
                            }
                        }
                    else:
                        window_parameters_file[idx] = {
                            'window_info': window_key,
                            'edit_parameters': {
                                'speculation_accuracy': val.speculation_accuracy,
                                'speculation_time': val.speculation_time,
                                'decoding_time_fn': val.decoding_time_fn_str # decoding_latency_fn_str
                            }
                        }
                    # print("decoding latency fn str", decoding_latency_fn_str)
                    # print("decoding latency fn str split", decoding_latency_fn_str.split("=")[1][1:-1])
                    window_metadata_file_ro.append(str(val))

                instruction_metadata_file_ro = []
                for idx, val in enumerate(self._device_manager.schedule_instructions):
                    instruction_metadata_file_ro.append(str(val))

                with open("window_parameters_file.json", "w", encoding="utf-8") as f:
                    json.dump(window_parameters_file, f, indent=2)

                with open("window_metadata_file_ro.json", "w", encoding="utf-8") as f:
                    json.dump(window_metadata_file_ro, f, indent=2)

                with open("instruction_metadata_file_ro.json", "w", encoding="utf-8") as f:
                    json.dump(instruction_metadata_file_ro, f, indent=2)

        return self.get_data(), self._window_manager.all_windows, self._window_manager.window_dag
    
    def initialize_experiment(
            self,
            schedule: LatticeSurgerySchedule,
            distance: int,
            scheduling_method: str,
            decoding_latency_fn: Callable[[int], int],
            speculation_mode: str | None,
            speculation_latency: int,
            speculation_accuracy: float,
            poison_policy: str = 'successors',
            missed_speculation_modifier: float = 1.4,
            max_parallel_processes: int | None = None,
            lightweight_setting: int = 0,
            rng: int | np.random.Generator = np.random.default_rng(),
            full_window_dag: nx.DiGraph | None = None,
            decoder_parameters: str | None = None,
            window_parameters: str | dict | None = None,
            slow_gate: bool = False,
            slow_gate_mult: int = None,
            spec_strat: str = "default",
            max_depth: int = None,
            spec_depth_threshold: int = None,
            **simulation_params,
        ) -> None:
        # get source code of the decoding latency function
        try:
            decoding_latency_fn_str = inspect.getsource(decoding_latency_fn)
        except Exception as e:
            print(f'Failed to get source of decoding_latency_fn: {e}')
            decoding_latency_fn_str = None

        # create the simulation parameters 
        self.simulation_params = SimulatorParams(
            distance=distance,
            scheduling_method=scheduling_method,
            decoding_latency_fn=decoding_latency_fn_str,
            speculation_mode=speculation_mode,
            speculation_latency=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            poison_policy=poison_policy,
            missed_speculation_modifier=missed_speculation_modifier,
            max_parallel_processes=max_parallel_processes,
            lightweight_setting=lightweight_setting,
            rng=(rng if isinstance(rng, int) else None),
            **simulation_params, # unpacks all key-val pairs from dict simulation_params and passes them as additional keyword args to SimulatorParams
        )

        # take in the json file for decoder parameters (w/ varying accuracies) and load the file
        decoder_parameters_dict = None
        if decoder_parameters is not None:
            path = Path(decoder_parameters)
            with path.open("r", encoding="utf-8") as f:
                decoder_parameters_dict =  json.load(f)

        # convert the window parameters to a dict if needed
        window_parameters_dict = None
        if window_parameters is not None:
            if isinstance(window_parameters, str):
                with open(window_parameters, "r", encoding="utf-8") as f:
                    window_parameters_dict = json.load(f)
            else:
                window_parameters_dict = window_parameters

        # print("window params dict simulator", window_parameters_dict)

        # create all the managers (distance here refers to code distance d)
        self.failed = False
        self._device_manager = DeviceManager(distance, schedule, lightweight_setting=lightweight_setting, rng=rng, slow_gate=slow_gate, slow_gate_mult=slow_gate_mult)
        if scheduling_method == 'sliding':
            self._window_manager = SlidingWindowManager(WindowBuilder(distance, lightweight_setting=lightweight_setting, decoder_parameters=decoder_parameters_dict, window_parameters=window_parameters_dict, schedule_insts=self._device_manager.schedule_instructions, speculation_accuracy=speculation_accuracy, slow_gate=slow_gate, slow_gate_mult=slow_gate_mult), lightweight_setting=lightweight_setting, window_parameters=window_parameters_dict)
        elif scheduling_method == 'parallel':
            self._window_manager = ParallelWindowManager(WindowBuilder(distance, lightweight_setting=lightweight_setting, decoder_parameters=decoder_parameters_dict, window_parameters=window_parameters_dict, schedule_insts=self._device_manager.schedule_instructions, speculation_accuracy=speculation_accuracy, slow_gate=slow_gate, slow_gate_mult=slow_gate_mult), lightweight_setting=lightweight_setting)
        elif scheduling_method == 'aligned':
            self._window_manager = TAlignedWindowManager(WindowBuilder(distance, lightweight_setting=lightweight_setting, decoder_parameters=decoder_parameters_dict, schedule_insts=self._device_manager.schedule_instructions, speculation_accuracy=speculation_accuracy, slow_gate=slow_gate, slow_gate_mult=slow_gate_mult), lightweight_setting=lightweight_setting)
        else:
            raise ValueError(f"Unknown scheduling method: {scheduling_method}")
        self._decoding_manager = DecoderManager(
            instruction_idx_dag=schedule.to_dag(),
            decoding_time_function=decoding_latency_fn,
            speculation_time=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            max_parallel_processes=max_parallel_processes,
            speculation_mode=speculation_mode,
            poison_policy=poison_policy,
            missed_speculation_modifier=missed_speculation_modifier,
            lightweight_setting=lightweight_setting,
            rng=rng,
            instructions=self._device_manager.get_instructions(),
            full_window_dag=full_window_dag,
            spec_strat=spec_strat,
            max_depth=max_depth,
            spec_depth_threshold=spec_depth_threshold,
        )

        # print(self._device_manager.schedule)

        self.start_time = dt.datetime.now()
        self.last_print_time = dt.datetime.now() - dt.timedelta(days=1) # not quite sure why we're subtracting 1 day from it
        print(self._device_manager.schedule)

    # def generate_full_window_dag(self):
    #     syndrome_rounds = self._device_manager.get_next_round(incomplete_instructions) 

    def step_experiment(self, pending_window_count_cutoff: int = 0, device_rounds_cutoff: int = 0, print_interval: dt.timedelta | None = None) -> None:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")

        if self.is_done():
            raise ValueError("Experiment is already done. Run run() to start a new experiment.")

        # how many windows are still pending/# windows that we still need to decode
        pending_window_count = len(self._window_manager.all_windows) - self._decoding_manager._num_completed_windows
        # print("complted windows", self._decoding_manager._num_completed_windows)
        # check if we hit cutoffs; if so, we failed -- exit
        if pending_window_count_cutoff > 0 and pending_window_count > pending_window_count_cutoff:
            self.failed = True
            return
        if device_rounds_cutoff > 0 and self._device_manager.current_round > device_rounds_cutoff:
            self.failed = True
            return

        # step device forward
        completed_window_indices = self._decoding_manager.step() # step decoding and speculation forward by 1 round
        purged_indices = self._window_manager.purge_windows(completed_window_indices) # in the code this does nothing?
        # | = set union in Python
        # incomplete instructions consist of: all unique active instructions (insts with active windows), all incomplete instructions that are waiting (have windows that are waiting), pending instruction indices (set of insts that are currently generating windows), and incmplete instruction indices (not yet decoded insts)
        incomplete_instructions = set(self._device_manager._active_instructions.keys()) | self._window_manager.window_builder.get_incomplete_instructions() | self._window_manager.pending_instruction_indices() | self._decoding_manager.get_incomplete_instruction_indices()
        # print("all incomplete instructions", incomplete_instructions)
        # print("all active insts", self._device_manager._active_instructions.keys())
        # print("all window waiting incmoplete", self._window_manager.window_builder.get_incomplete_instructions())
        # print("all pending insts", self._window_manager.pending_instruction_indices() )
        # print("all incomplete instructions decoding manager", self._decoding_manager.get_incomplete_instruction_indices())

        # print("all windows ", len(self._window_manager.all_windows), " ", self._window_manager.all_windows)
        # print("Nodes:")
        # for n, attrs in self._window_manager.window_dag.nodes(data=True):
        #     print(f"  {n}: {attrs}")

        # print("\nEdges:")
        # for u, v, attrs in self._window_manager.window_dag.edges(data=True):
        #     print(f"  {u} -> {v}: {attrs}")

        syndrome_rounds = self._device_manager.get_next_round(incomplete_instructions) # returns another round of syndrome measurements, starting new insts if possible
        # print("incomplete instructions ", incomplete_instructions)
        # print("syndrome rounds simulator", syndrome_rounds)
        
        cur_time = dt.datetime.now()
        # print update at set interval
        if print_interval is not None and cur_time - self.last_print_time >= print_interval:
            num_complete_instructions = self._device_manager._completed_instruction_count
            print(f'{cur_time.strftime("%Y-%m-%d %H:%M:%S")} | Simulation update: decoder round {self._decoding_manager._current_round}, completed instructions: {num_complete_instructions}/{len(self._device_manager.schedule)}, actively running or decoding instructions: {len(incomplete_instructions)}, waiting windows: {pending_window_count}/{len(self._window_manager.all_windows)}. Max active instruction index: {max(incomplete_instructions, default=-1)}') # this max fcn returns the largest value in the incomplete_instructions set
            sys.stdout.flush()
            self.last_print_time = cur_time
            
        # process new round
        # window manager processes new round of syndrome measurements, gets new windows from this new round of measurements
        newly_constructed_windows = self._window_manager.process_round(syndrome_rounds)
        # print("newly constructed windows simulator", newly_constructed_windows)
        # decoding manager now updates decoding with these new windows and new window dag
        self._decoding_manager.update_decoding(newly_constructed_windows, purged_indices, self._window_manager.window_dag)

        

        # print("current round", self._decoding_manager._current_round)
        # print("all active insts", self._device_manager._active_instructions)
        # print("all active windows", self._decoding_manager._active_window_progress)
        # print("all speculating windows", self._decoding_manager._active_speculation_progress)

    # Done when we either fail, or we have completed decoding every single window
    def is_done(self) -> bool:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        return self.failed or (self._device_manager.is_done() and len(self._window_manager.all_constructed_windows) - self._decoding_manager._num_completed_windows == 0)

    # returns 1) boolean on whether we succeeded in simulation or not 2) the simulation's parameters
    # 3) the data of the device history 4) window information like the graph and volume and windows and edges
    # 5) decoding data (e.g. # missed speculations, window decoding start/end times, max parallel decoders, etc)
    def get_data(self) -> tuple[bool, SimulatorParams, DeviceData, WindowData, DecoderData]:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        device_data = self._device_manager.get_data()
        window_data = self._window_manager.get_data()
        decoding_data = self._decoding_manager.get_data()
        return not self.failed, self.simulation_params, device_data, window_data, decoding_data
    
    # get data of the current frame we're on?
    def get_frame_data(self) -> list[plt.Axes]:
        return self.frame_data