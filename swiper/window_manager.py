"""

Ideas for performance optimization (if needed later):
- window_idx_dict: dict[DecodingWindow, int] to avoid calling index() on all_windows
- Priority queue for buffer wait
"""
from abc import ABC, abstractmethod
import networkx as nx
from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray

from swiper.window_builder import WindowBuilder, DecodingWindow, SpacetimeRegion
from swiper.device_manager import SyndromeRound
from swiper.lattice_surgery_schedule import Instruction

@dataclass
class WindowData:
    """Data structure to hold information about windows throughout device run.
    """
    all_windows: list[DecodingWindow | None] # every decoding window that's every been creatged durign the simulation
    all_constructed_windows: list[int] # indices of windows that have been fully constructed (complete and ready for decoding, no more dependencies). decoding engine will only decode windows once they appear in this list
    window_dag_edges: list[tuple[int, int]] # represents dag of dependencies between windows (each tuple (a,b) means that window a must finish b4 window b can start being decoded)
    window_construction_times: dict[int, int] # map the constructed window's index to the round number when it was marked as constructed
    window_volumes: list[tuple[list[int], list[int]]] # list of (commit_region_durations, buffer_region_durations) (captures spacetime volume of each window) -- commit_region_durations is a list of the duration of every commit region within that window

    def to_dict(self):
        return asdict(self)
    
    # get the window at a specific index
    def get_window(self, window_idx: int) -> DecodingWindow:
        window = self.all_windows[window_idx]
        if window is None:
            raise ValueError("Window does not exist")
        return window

class WindowManager(ABC):
    
    def __init__(self, window_builder: WindowBuilder, lightweight_setting: int = 0, window_parameters: dict | None = None):
        self.all_windows: list[DecodingWindow | None] = [] # a list of every decoding window that's created (each window's idx essentially serves as its ID)
        self.all_constructed_windows: list[int] = [] # stores the indices of all fully constructed windows, meaning these windows are ready to be decoded (commit rgns complete, no dangling boundaries)
        self.window_builder = window_builder # window builder, to help build new windows with complete commit regions after some number of rounds
        self.window_dag = nx.DiGraph() # dag of the windows
        self.window_end_lookup: dict[tuple[tuple[int, int], int], tuple[int, int]] = {} # lookup table to find which window ends at a given spacetime point. key = ((patch_id_x, patch_id_y), end_round), val = (window_idx, commit_rgn_idx)
        self.window_future_buffer_wait: dict[int, int] = {} # tracks windows waiting for future buffer rgns to arrive b4 they can be marked as constructed. key=window idx, val=# of remaining buffer steps they must wait for (each simulation round, counter is decremented. once counter=0, window is rdy to finalize)
        self.window_construction_wait: set[int] = set() # indices of windows that are not fully surrounded (missing spatial or temporal neighbors) --> candidates for construction, checked each round to see if they've become ready
        self.current_round = 0 # current syndrome round number
        self._window_construction_times: dict[int, int] = {} # records when each window was marked constructed. key=window idx, val=round num when window was marked constructed
        self._unconstructed_window_indices: dict[DecodingWindow, int] = {} # maps active, unconstructed window objects to their index in self.all_windows (limited to windows under construction)
        self.lightweight_setting = lightweight_setting # lightweight setting, affects how much info (simulation data) we store
        self.window_parameters = window_parameters

        # added to delay construction
        self._constructed_windows_delay: dict[int, int] = {} # key = window idx, val = how much longer (in rounds) until we can decode the window

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        """Process new syndrome rounds and update the decoding window dependency
        graph as needed.
        """
        raise NotImplementedError
    
    # get the window for a certain window index
    def _get_window(self, window_idx: int) -> DecodingWindow:
        window = self.all_windows[window_idx]
        if window is None:
            raise ValueError("Window does not exist")
        return window

    # get the instruction indices that are currently generating windows
    def pending_instruction_indices(self) -> set[int]:
        """Get the set of instruction indices that are currently generating
        windows.
        """
        # get all windows in the unconstructed window indices dict (keys are the DecodingWindow objs), and for each window get its parent instr index, which is all instruction indexes that affected this window
        return set([instr_idx for window in self._unconstructed_window_indices.keys() for instr_idx in window.parent_instr_idx])

    # get all unconstructed window's indexes
    def get_unconstructed_windows(self):
        return list(self._unconstructed_window_indices.values())

    # remove a window from all datakeeping structures
    def _remove_window(self, window_idx: int) -> None:
        """TODO"""
        # Remove window from all_windows, and update window indices
        window = self._get_window(window_idx) # get the window of a specific window_idx
        if window in self._unconstructed_window_indices: # if window is unconstructed, then we remove this window from the unconstructed window list
            self._unconstructed_window_indices.pop(window)
        self.all_windows[window_idx] = None # set this window index in all_windows to None (bc we remove the window)
        self.window_dag.remove_node(window_idx) # remove the node in the dag for this window idx
        # if this window index is in future_buffer_wait or window_construction_wait, remove it
        if window_idx in self.window_future_buffer_wait:
            self.window_future_buffer_wait.pop(window_idx)
        if window_idx in self.window_construction_wait:
            self.window_construction_wait.remove(window_idx)
        for cr in window.commit_region: # go thru all commit regions in this window
            if (cr.patch, cr.round_start + cr.duration) in self.window_end_lookup: # if this commit region's patch and end round is in window_end_lookup
                if self.window_end_lookup[(cr.patch, cr.round_start + cr.duration)][0] == window_idx: # we lookup this commit region and check if its window idx is the same as our current window's idx
                    self.window_end_lookup.pop((cr.patch, cr.round_start + cr.duration)) # if so, we get rid of this (because we're trying to remove this window from every bookeeping data structure)
        del window

    def _all_regions_touching(self, regions: list[SpacetimeRegion]):
        connectivity = []
        for i,cr1 in enumerate(regions): # go through all SpacetimeRegions in list
            for j,cr2 in enumerate(regions):
                if i != j and cr1.shares_boundary(cr2): # if any 2 of the SpacetimeRegions in the list share a boundary (temporal or spatial)
                    connectivity.append((i,j)) # we append these 2 regions to our connectivity list
        g = nx.Graph()
        g.add_nodes_from(range(len(regions))) # nodes are each region in regions list
        g.add_edges_from(connectivity) # edges of graph are based on connectivity btwn regions
        return nx.is_connected(g) # use this graph to check whether all the regions are touching/connected with each other (checks if the entire undirected graph forms a single connected component)
    
    def _get_touching_unconstructed_window_indices(self, window: DecodingWindow) -> list[int]:
        """Get all unconstructed windows that share a boundary with window.
        """
        adjacent_windows = []
        # check if this window shares a boundary with any of the unconstructed windows, if so, append the unconstructed window's index to the list
        for other_window, other_idx in self._unconstructed_window_indices.items():
            if window.shares_boundary(other_window):
                adjacent_windows.append(other_idx)
        return adjacent_windows

    def _merge_windows(
            self,
            window_1: DecodingWindow,
            window_2: DecodingWindow,
            enforce_contiguous: bool = True,
        ) -> DecodingWindow:
        """Merge two windows into a new window. Removes window_2 from
        all_windows.

        WARNING: like _append_to_buffers or _mark_constructed, this method
        modifies all_windows, window_dag, window_buffer_wait, and other internal
        data structures.

        After merging, the entry for window_1 in self.all_windows will be
        updated to contain the combined commit region and buffer regions.
        window_2 will be removed from self.all_windows.

        Args:
            window_1: Main window
            window_2: Window to merge into window_1
            enforce_contiguous: If True, the commit regions of window_1 and
                window_2 must be contiguous in spacetime.
        
        Returns:
            new_window: The new window created by merging window_1 and window_2.
                Note that this window will already be in self.all_windows.
        """
        assert window_1.constructed == window_2.constructed == False # assert that none of these windows are finished being constructed with buffers
        # get the indices of our 2 windows
        window_idx_1 = self._unconstructed_window_indices[window_1] 
        window_idx_2 = self._unconstructed_window_indices[window_2]

        # enforce contiguous means that all commit regions of both windows must be contiguous/touching, so we enforce this using the if-statement
        if enforce_contiguous:
            if not self._all_regions_touching(list(window_1.commit_region) + list(window_2.commit_region)):
                raise ValueError("Commit regions must be contiguous")

        # Add window 2's attributes to window 1
        for succ in self.window_dag.successors(window_idx_2): # add edge btwn window_idx_1 and all of window 2's successors in dag
            if succ != window_idx_1:
                self.window_dag.add_edge(window_idx_1, succ)
        for pred in self.window_dag.predecessors(window_idx_2): # add edge btwn window_idx_1 and all of window_idx_2's predecessors 
            if pred != window_idx_1:
                self.window_dag.add_edge(pred, window_idx_1)
        if window_idx_2 in self.window_construction_wait: # if window_idx_2 is in this list, which means that its surrounding windows are not yet constructed
            self.window_construction_wait.add(window_idx_1) # we add window_idx_1 to this list, since window_idx_1 now has these surrounding windows that are not yet constructed bc window 1 is merged with window 2
        # for every SpaceTimeRegion in the commit region, we check if this rgn is in window_end_lookup, and get the corresopnding window that ends at this rgn's endpoint
        for cr_idx,cr in enumerate(window_2.commit_region): 
            key = (cr.patch, cr.round_start + cr.duration)
            if key in self.window_end_lookup:
                if self.window_end_lookup[key][0] == window_idx_2: # if the end of this commit rgn patch is equal to the second window index
                    self.window_end_lookup[key] = (window_idx_1, len(window_1.commit_region)+cr_idx) # we change it to window idx 1, with new commit region index based on length of window 1 commit region (because now window 2 should be merged with window 1)
        # for k,(w_idx,cr_idx) in self.window_end_lookup.items(): # TODO: make this not scale as O(n)
        #     if w_idx == window_idx_2:
        #         self.window_end_lookup[k] = (window_idx_1, len(window_1.commit_region)+cr_idx)

        # Construct new window and replace window_1 in all_windows
        new_window = DecodingWindow( # create new window which merges window 1 and window 2
            commit_region=tuple(list(window_1.commit_region) + list(window_2.commit_region)),
            buffer_regions=window_1.buffer_regions | window_2.buffer_regions,
            merge_instr=window_1.merge_instr | window_2.merge_instr,
            parent_instr_idx=window_1.parent_instr_idx | window_2.parent_instr_idx,
            window_idx=window_idx_1,
            constructed=False,
            speculation_accuracy=window_1.speculation_accuracy, # TODO change -- when we merge 2 windows how should the speculation accuracy change/
            speculation_time=window_1.speculation_time,
            decoding_time_fn=window_1.decoding_time_fn,
            decoding_time_fn_str=window_1.decoding_time_fn_str,
        )
        self.all_windows[window_idx_1] = new_window # replace window_idx_1 w this new DecodingWindow object
        self._unconstructed_window_indices.pop(window_1) # remove the original window 1 object from unconstructed_window_indices (key=DecodingWindow obj here)
        self._unconstructed_window_indices[new_window] = window_idx_1 # create new object in unconstructed_window_indices with merged iwndow

        self._remove_window(window_idx_2) # remove window_idx_2 from all bookeeping structures

        return new_window # return this new window

    def _append_to_buffers(self, window: DecodingWindow, region: SpacetimeRegion) -> DecodingWindow:
        """Create new window with region appended to buffer regions
        """
        assert not window.constructed # ensure window not already fully constructed
        if region in window.buffer_regions: # if region is alr in window's buffer regions, can return
            return window
        new_window = DecodingWindow( # create new DecodingWindow object, with same everything except add this new region to the window's buffer_regions
            commit_region=window.commit_region, 
            buffer_regions=window.buffer_regions | frozenset([region]), # add new rgn to window's buffer rgns
            merge_instr=window.merge_instr, 
            parent_instr_idx=window.parent_instr_idx,
            window_idx=window.window_idx,
            constructed=False,
            speculation_accuracy=window.speculation_accuracy,
            speculation_time=window.speculation_time,
            decoding_time_fn=window.decoding_time_fn,
            decoding_time_fn_str=window.decoding_time_fn_str,
        )
        window_idx = self._unconstructed_window_indices[window] # get the window_idx of this window
        self.all_windows[window_idx] = new_window # replace window_idx's value in all_windows with this new window
        # replace unconsturcted_window_indices (key,val) with new window
        self._unconstructed_window_indices.pop(window)
        self._unconstructed_window_indices[new_window] = window_idx
        del window # delete old window
        return new_window # return new window
    
    # CURR WINDOW MUST CONTAIN ENTIRE THING, NOT JUST WINDOW_INFO
    # BECAUSE LATER WHEN RETURN NEED TO RETURN THE EDIT PARAMS TOO SO THAT WE CAN ACCESS IT IN MARK_CONSTRUCTED
    # make sure that mark_constructed is the right place to put these edited parameters
    def window_in_config(self, window: DecodingWindow):
        if self.window_parameters is not None:
            # print("len of window params", len(self.window_parameters))
            same_inst_windows = {}
            same_inst_and_commit_windows = {}

            # search through all windows in our config file
            # first narrow down search space by parent inst index
            # create a list of all window dicts (window_info dicts) that have 
            for idx, val in self.window_parameters.items(): # curr_window is a dict
                curr_window = val['window_info'] # get the window info of the current window we're looking at
                if window.parent_instr_idx == frozenset(curr_window['parent_instr_idx']):
                    # same_inst_windows.append({idx: curr_window})
                    same_inst_windows[idx] = val # curr_window

            # then from this narrowed down list of windows (contains window_info dicts), 
            # search these windows for windows with matching commit regions
            # create a list of all window dicts (window_info dicts) that have the same commit region (in addition to already same parent inst)
            for idx, val in same_inst_windows.items():
                curr_window = val['window_info']
                if len(curr_window['commit_region']) > 1 or len(window.commit_region) > 1: # we should only have one commit region
                    raise ValueError(f"Commit Region Not Length 1: {len(curr_window['commit_region'])} {len(window.commit_region)}")
                
                if (tuple(curr_window['commit_region'][0]['patch']) == window.commit_region[0].patch 
                    and curr_window['commit_region'][0]['duration'] == window.commit_region[0].duration
                    and curr_window['commit_region'][0]['num_spatial_boundaries'] == window.commit_region[0].num_spatial_boundaries
                    and curr_window['commit_region'][0]['initialized_patch'] == window.commit_region[0].initialized_patch
                    and curr_window['commit_region'][0]['discard_after'] == window.commit_region[0].discard_after):
                    same_inst_and_commit_windows[idx] = val

            # now from same commit rgn and parent insts windows, we want to search for if they have the same buffer region
            # because buffer region can have multiple spacetime regions, need to compare their frozensets (a bit more hard)
            for idx, val in same_inst_and_commit_windows.items():
                curr_window = val['window_info']
                window_obj_regions = {(o.patch, o.duration, o.num_spatial_boundaries, o.initialized_patch, o.discard_after) for o in window.buffer_regions}
                window_dict_regions = {(tuple(d['patch']), d['duration'], d['num_spatial_boundaries'], d['initialized_patch'], d['discard_after']) for d in curr_window['buffer_regions']}
                if window_obj_regions == window_dict_regions:
                    # print("MATCH FOUND")
                    # remove this match from the window parameters dict
                    del self.window_parameters[idx]
                    return val
                
        return None # either no window parameters, or didn't find a match for the current window in window parameters

    def _mark_constructed(self, window_idx: int) -> DecodingWindow:
        """Create new window marked constructed to indicate it is ready to be
        decoded.
        """
        window = self._get_window(window_idx) # get the window at this index
        assert not window.constructed # assert it's not constructed yet
        # this window_idx is no longer waiting for any future buffer rgns
        if window_idx in self.window_future_buffer_wait:
            self.window_future_buffer_wait.pop(window_idx)
        # this window_idx is no longer waiting for any of its surrounding regions to be constructed
        if window_idx in self.window_construction_wait:
            self.window_construction_wait.remove(window_idx)

        # self._constructed_windows_delay[window_idx] = 0 # TODO: set to whatever delay you want -- I'm just going to set to 7 for now
        # print("delay constructed windows in mark constructed", self._constructed_windows_delay)
        # print("window parent idx", window_idx, window.parent_instr_idx)

        self.all_constructed_windows.append(window_idx) # append this window idx to constructed windows list
        self._window_construction_times[window_idx] = self.current_round # append the current round as this window's construction time to the list
        new_window = DecodingWindow( # create new window with this window's same metadata, except now constructed=True
            commit_region=window.commit_region,
            buffer_regions=window.buffer_regions,
            merge_instr=window.merge_instr,
            parent_instr_idx=window.parent_instr_idx,
            window_idx=window.window_idx,
            constructed=True, # this is the only thing that changed!
            speculation_accuracy=window.speculation_accuracy,
            speculation_time=window.speculation_time,
            decoding_time_fn=window.decoding_time_fn,
            decoding_time_fn_str=window.decoding_time_fn_str,
        )
        config_window = self.window_in_config(new_window)
        if config_window is not None:
            new_window = DecodingWindow( # create new window with this window's same metadata, except now constructed=True
                commit_region=window.commit_region,
                buffer_regions=window.buffer_regions,
                merge_instr=window.merge_instr,
                parent_instr_idx=window.parent_instr_idx,
                window_idx=window.window_idx,
                constructed=True, # this is the only thing that changed!
                speculation_accuracy=config_window['edit_parameters']['speculation_accuracy'],
                speculation_time=config_window['edit_parameters']['speculation_time'],
                decoding_time_fn=eval(config_window['edit_parameters']['decoding_time_fn']),
                decoding_time_fn_str=config_window['edit_parameters']['decoding_time_fn'],
            )
        self.all_windows[window_idx] = new_window # update all_windows with this new window
        self._unconstructed_window_indices.pop(window) # delete this window from unconstructed_window_indices dict
        del window
        return new_window
    
    def purge_windows(self, window_indices) -> None: # this fcn does nothing? just returns immediately
        return
        for window_idx in window_indices:
            window = self.all_windows[window_idx]
            if window:
                for cr in window.commit_region:
                    if (cr.patch, cr.round_start + cr.duration) in self.window_end_lookup:
                        if self.window_end_lookup[(cr.patch, cr.round_start + cr.duration)][0] == window_idx:
                            self.window_end_lookup.pop((cr.patch, cr.round_start + cr.duration))
                self.all_windows[window_idx] = None
                self.window_dag.remove_node(window_idx)
                pass
    
    def count_covered_faces(self, window_idx, cr_idx):
        # Need to make sure every face of every commit region either touches
        # another commit region, has an outgoing buffer, has an incoming
        # buffer, or is an initialization/discard/spatial boundary.
        window = self._get_window(window_idx) # get window at this idx
        cr = window.commit_region[cr_idx] # get this window's commit rgn at the specified index
        predecessors = list(self.window_dag.predecessors(window_idx)) # get all the predecessors of this window

        num_commit_neighbors = 0 # number of this window's cmt rgns that this cmt rgn shares a boundary with
        num_incoming_other_buffers = 0
        num_outgoing_buffers = 0 # number of outgoing buffers that this commit rgn touches
        num_terminations = 0
        for cr1 in window.commit_region:
            # check all commit regions in this window that share a boundary with the current commit rgn we're looking at
            if cr1 != cr and cr1.shares_boundary(cr):
                num_commit_neighbors += 1
        for other_idx in predecessors: # iterate throuhg all predecessor windows
            other_window = self._get_window(other_idx)
            # check all predecessor's buffer rgns and see if they overlap with our current cmt rgn -- if so, we add to num_incoming_other_buffers the number of rgns that touch this buffer rgn (add len(commits) bc single buffer boundary can touch multiple commit faces TODO??)
            for br,commits in other_window.buffer_boundary_commits().items(): # buffer_boundary_commits gives all commit rgns that this other window's buffer rgn shares a boundary with (key=buffer rgn, c=cmt rgn)
                if br.overlaps(cr): # if the buffer rgn overlaps with the commit rgn
                    num_incoming_other_buffers += len(commits) # add to number of incoming other buffers the number of commit rgns that this buffer rgn touches
        for br,commits in window.buffer_boundary_commits().items(): # iterate throuhg this window's buffer_boundary_commits
            if cr in commits: # if the commit rgn we're looking at is touching this buffer rgn
                num_outgoing_buffers += 1 # add to num_outgoing_buffers
        # if commit rgn is an initialized patch, has a prior_t, or is discard_after, add to num_terminations
        if cr.initialized_patch or cr.prior_t:
            num_terminations += 1
        if cr.discard_after:
            num_terminations += 1
        num_terminations += cr.num_spatial_boundaries # add to num_terminations the number of spatial boundaries of this commit rgn too

        return num_commit_neighbors, num_incoming_other_buffers, num_outgoing_buffers, num_terminations
    
    def _update_waiting_windows(self) -> None:
        """Look for dangling windows to mark as constructed.
        """
        constructed_windows = set()
        for window_idx in self.window_future_buffer_wait.keys(): # this window is waiting for future buffer rgns to arrive
            self.window_future_buffer_wait[window_idx] -= 1 # decrement # rounds waiting by 1
            if self.window_future_buffer_wait[window_idx] <= 0: # if this window is now waiting 0 more rounds, we add it to constructed windows (bc now this window can be constructed)
                constructed_windows.add(self._get_window(window_idx))

        surrounded_windows = set()
        for window_idx in self.window_construction_wait: # if this window is waiting for its surrounding windows to be constructed
            # Need to make sure every face of every commit region either touches
            # another commit region, has an outgoing buffer, has an incoming
            # buffer, or is an initialization/discard/spatial boundary.
            window = self._get_window(window_idx)
            ready_to_construct = True
            for cr_idx in range(len(window.commit_region)): # check all of this window's commit regions (each one is a SpacetimeRegion)
                num_commit_neighbors, num_incoming_other_buffers, num_outgoing_buffers, num_terminations = self.count_covered_faces(window_idx, cr_idx) # get the number of covered faces for this commit rgn and window
                total = num_commit_neighbors + num_incoming_other_buffers + num_outgoing_buffers + num_terminations # total number of covered faces/faces that this window borders
                # print("total", total, num_commit_neighbors, num_incoming_other_buffers, num_outgoing_buffers, num_terminations)
                # checks that all of the faces of this cmt rgn either touches another cmt rgn, has an outgoing buffer, incoming buffer, or is init/discard/spatial boundary. 
                # total=6 means that this is satisfied. total=# faces that satisfy this
                if total < 6: 
                    ready_to_construct = False
                    break

                if total != 6: # this is for total > 6, which shouldn't happen
                    cr = window.commit_region[cr_idx]
                    raise ValueError(f'Total should be 6, but is {total}. {num_commit_neighbors}, {num_incoming_other_buffers}, {num_outgoing_buffers}, {num_terminations}, {cr.num_spatial_boundaries}, {window}, {cr}')
                assert total == 6 # total shld be 6
            if ready_to_construct: # if ready to construct, add to surrounded windows
                surrounded_windows.add(window_idx)
            
        # original
        for window_idx in surrounded_windows: # for windows in surrounded_windows, mark them as constructed
            self._mark_constructed(window_idx)

    def _flush_windows(self) -> None:
        # print("in flush windows")
        # No new rounds; flush any dangling windows
        unconstructed_windows = list(self.window_construction_wait) # get all unconstructed windows who are waiting for their surrounding windows to be constructed
        for window_idx in unconstructed_windows:
            window = self._get_window(window_idx)
            assert not window.constructed # assert that this window is not constructed
            self._mark_constructed(window_idx) # mark this window as constructed, since we'll have no more new rounds anyways
    
    def _clean_old_windows(self, newly_constructed_window_indices) -> None: 
        # Remove old windows from all_windows
        assert self.lightweight_setting >= 1 # assert lightweight setting (only want to clean if our lightweight setting is this)
        for window_idx in newly_constructed_window_indices: # for all newly constructed windows
            for neighbor_idx in set(self.window_dag.successors(window_idx)) | set(self.window_dag.predecessors(window_idx)): # get set of successors and predecessors of this window as neighbors
                if self.all_windows[neighbor_idx]: # if this neighbor index is not None in all_windows
                    can_clean = True # means we should clean
                    # get all the neighbors (successors and predecessors) of the neighbor
                    for neighbor_neighbor_idx in set(self.window_dag.successors(neighbor_idx)) | set(self.window_dag.predecessors(neighbor_idx)):
                        # if the neighbor window borders a newly constructed window, don't want to clean it yet
                        if neighbor_neighbor_idx in newly_constructed_window_indices: # if the neighbor of the neighbor is a newly constructed window
                            can_clean = False # then we shouldn't clean this window
                            break
                    if can_clean: # if we can clean this window, then we want to remove this object from all_windows
                        self.all_windows[neighbor_idx] = None
                # neighbor_window = self.all_windows[neighbor_idx]
                # if neighbor_window is not None and not neighbor_window.constructed:
                #     can_clean = False
                #     break
            # if can_clean:
            #     self.all_windows[window_idx] = None

    def get_data(self) -> WindowData: # return different amounts of data, depending on what lightweight setting we have
        if self.lightweight_setting == 0:
            return WindowData(
                all_windows=self.all_windows, # return all windows
                all_constructed_windows=self.all_constructed_windows, # return all constructed windows' indices
                window_dag_edges=list(self.window_dag.edges), # returns edges of the window dag
                window_construction_times=self._window_construction_times, # returns window construction times
                # window volumes returns the duration for all commit and buffer SpaceTime regions of the window, for all constructed windows
                window_volumes=[([cr.duration for cr in window.commit_region], [br.duration for br in window.buffer_regions]) for window in [self._get_window(window_idx) for window_idx in self.all_constructed_windows]]
            )
        elif self.lightweight_setting == 1: # only return window construction time and volumes here
            return WindowData(
                all_windows=None,
                all_constructed_windows=None,
                window_dag_edges=None,
                window_construction_times=self._window_construction_times,
                window_volumes=[([cr.duration for cr in window.commit_region], [br.duration for br in window.buffer_regions]) for window in [self._get_window(window_idx) for window_idx in self.all_constructed_windows]]
            )
        elif self.lightweight_setting == 2 or self.lightweight_setting == 3: # return none of this data
            return WindowData(
                all_windows=None,
                all_constructed_windows=None,
                window_dag_edges=None,
                window_construction_times=None,
                window_volumes=None
            )
        else:
            raise ValueError('Invalid lightweight setting')

# sliding window manager strategy!
class SlidingWindowManager(WindowManager):
    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        constructed_window_count = len(self.all_constructed_windows) # get num of constructed windows
        new_rounds_copy = new_rounds.copy()
        new_rounds_dummy = new_rounds.copy()
        new_rounds_dummy[:] = [r for r in new_rounds if r.instruction_idx == -2]
        new_rounds[:] = [r for r in new_rounds if r.instruction_idx != -2]
        # print("new_rounds", new_rounds)
        # print("new rounds dummy", new_rounds_dummy)
        # print("og new rounds", new_rounds_copy)

        # if we have new rounds, build the windows/update the existing windows for those new rounds
        # if new_rounds:
        if new_rounds_copy:
            # new_commits = self.window_builder.build_windows(new_rounds)
            new_commits = self.window_builder.build_windows(new_rounds_copy)
        else: # if no new rounds, flush remaining rounds into windows
            new_commits = self.window_builder.flush()

        # print("new commits", new_commits)

        # print("all constructed windows1 ", self.all_constructed_windows)
        
        if new_commits: # if we need to add new windows
            # Add new windows
            new_window_start = len(self.all_windows)
            self.all_windows.extend(new_commits) # add these new windows to the list of all windows
            self._unconstructed_window_indices.update({window: new_window_start+i for i,window in enumerate(new_commits)}) # add to unconstructed windows all of these new windows' indices (windows start as unconstructed)

            # enumerate through all new windows
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                # we take idx 0 because newly created windows are guaranteed to have exactly one commit region
                patch = window.commit_region[0].patch # get the patch of this window's commit region (we only look at the first )
                end = window.commit_region[0].round_start + window.commit_region[0].duration # get the end round of this window's commit region
                assert (patch, end) not in self.window_end_lookup # assert that this patch and end is not the end point of an already existing window
                self.window_end_lookup[(patch, end)] = (window_idx, 0) # add this patch, end pair to window_end_lookup, with window_idx and commit rgn idx (0 here bc we look at idx 0)
                self.window_dag.add_node(window_idx) # add window idx to the dag
                self.window_construction_wait.add(window_idx) # add this window to construction wait, meaning its waiting to be constructed
            # print("all constructed windows2 ", self.all_constructed_windows)
            
            # Process buffers in time (windows covering same patch)
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region[0].patch # get the patch of the window's commit region
                prev_window_end = window.commit_region[0].round_start # get the previous window's end, which is the curr window's commit region's start
                if (patch, prev_window_end) in self.window_end_lookup: # if this patch and previous window end is in window_end_lookup (is the endpoint of an existing window)
                    prev_window_idx,_ = self.window_end_lookup[(patch, prev_window_end)] # get the previous window index whose end is this one
                    prev_window = self._get_window(prev_window_idx) # get the previous window
                    prev_commit = prev_window.commit_region[0] # get the previous window's commit region
                    if prev_commit.discard_after: # if the previous window's commit region will be discarded after, just continue
                        continue
                    # For sliding window, buffers extend at most one step forward in time
                    # this means we only have to add to the previous window one commit region as the one buffer region of th eprevious window
                    assert not prev_window.constructed # assert that the previous window is not constructed yet
                    prev_window = self._append_to_buffers(prev_window, window.commit_region[0]) # create new prev_window, where prev_window is the main window, and window.commit_region[0] is the new buffer region of the previous window
                    self.window_dag.add_edge(prev_window_idx, window_idx) # add this dependency to the window dag, from the previous window to the current window
            # print("all constructed windows3 ", self.all_constructed_windows)

            # Process buffers in space (windows covering same MERGE instruction)
            unconstructed_window_indices = self.get_unconstructed_windows()
            # iterate through all unconstructed windows
            for window_idx in unconstructed_window_indices:
                window = self._get_window(window_idx)
                # generally unconstructed windows all have only 1 commit region
                cr = window.commit_region[0] # get the first commit rgn of these windows
                merge_instr = cr.merge_instr # get the merge inst of this commit region
                if merge_instr: # if it has a merge inst
                    touching_windows = self._get_touching_unconstructed_window_indices(window) # get all the touching unconstructed window indices of this window
                    for w_idx in touching_windows: # iterate through each of these bordering, unconstructed windows
                        other_window = self._get_window(w_idx) # get the other window
                        other_cr = other_window.commit_region[0] # get this other commit region
                        if other_cr.merge_instr == merge_instr: # if this other commit region's merge instruction is the same as ours (meaning the 2 regions are merged)
                            # get the patches of the 2 merged commit regions
                            patch1 = cr.patch
                            patch2 = other_window.commit_region[0].patch
                            # if these merged faces are stored, and patch1 is down-right of patch2 (enforces a consistent ordering for work so that we don't do double work)
                            if ((patch1, patch2) in merge_instr.merge_faces or (patch2, patch1) in merge_instr.merge_faces) and (patch1[0] >= patch2[0]) and (patch1[1] >= patch2[1]):
                                if other_cr not in window.buffer_regions: # if this other commit region is not alreday in the curr window's buffer regions
                                    assert not window.constructed and not other_window.constructed # assert both windows not done constructing
                                    window = self._append_to_buffers(window, other_window.commit_region[0]) # make this other window's commit region our curr window's buffer region (bc merged)
                                    cr = window.commit_region[0]
                                    self.window_dag.add_edge(window_idx, w_idx) # add the dependency btwn our window and the other touching window
            # print("all constructed windows4 ", self.all_constructed_windows)

        self._update_waiting_windows() # might need to mark as constructed some previously dangling windows bc we just updated windows
        # print("all constructed windows5 ", self.all_constructed_windows)

        if not new_rounds and not new_rounds_dummy: # flush windows if no new rounds # flush dangling windows (mark dangling windows as constructed, if no new rounds)
            self._flush_windows()

        # TODO added
        # for window_idx in list(self._constructed_windows_delay):
        #     self._constructed_windows_delay[window_idx] -= 1
        #     if self._constructed_windows_delay[window_idx] <= 0:
        #         del self._constructed_windows_delay[window_idx]
        #         self.all_constructed_windows.append(window_idx)

        # print("all constructed windows", self.all_constructed_windows)
        # print("delay constructed windows", self._constructed_windows_delay)

        # update constructed windows with our new windows (start iterating thru all_constructed_windows from the index where our new windows start)
        constructed_windows = [self._get_window(window_idx) for window_idx in self.all_constructed_windows[constructed_window_count:]]
        # print("all constructed windows7 ", self.all_constructed_windows)

        if self.lightweight_setting > 0: # if lightweight setting more than 0, clean our old windows (get rid of no-longer-needed data, pertaining to our newly constructed windows)
            self._clean_old_windows(self.all_constructed_windows[constructed_window_count:]) # clean our old windows
        # print("all constructed windows8 ", self.all_constructed_windows)

        self.current_round += 1 # iterate round
        return constructed_windows # return constructed windows
        
# parallel window manager prediction strategy
class ParallelWindowManager(WindowManager):
    """TODO
    
    Ideally, a source contains one commit region and some buffer regions
    surrounding it. A sink contains three commit regions in a line, where the
    start and end are buffered by neighboring sources. However, this gets
    complicated when we have dense merge schedules and more pipe junctions.

    Addition: there are two layers of sources. By default, we only use the
    second source layer and the sinks, but in a big merge operation, we can 
    """
    # summary above in other words: sink shld contain a bunch of commit rgns going into it, who are buffered by other sources. 
    # source shld contain only one commit rgn and some buffer rgns surrounding it (so that it can give its dependencies away)
    layer_indices: list[set[int]]

    def __init__(self, window_builder: WindowBuilder, lightweight_setting: int = 0):
        # layer idx 0 = alternate source (backup/auxiliary source layer, used when layer 1 full/not ideal)
        # layer idx 1 = main source
        # layer idx 2 = sink layer that receives data/commit regions from source
        self.layer_indices = [set(), set(), set()] # each set holds indices of windows that belong to a specific layer
        super().__init__(window_builder, lightweight_setting=lightweight_setting)

    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        constructed_window_count = len(self.all_constructed_windows) # get num of constructed windows

        if new_rounds: # if we have new SyndromeRounds to process, build windows for these rounds
            new_commits = self.window_builder.build_windows(new_rounds)
        else: # if no new SyndromeRounds to process, flush windows (handle remainder)
            new_commits = self.window_builder.flush()

        if new_commits: # if we have new windows that we have to process (new_commits is list[DecodingWindow])
            self._add_new_commits(new_commits) # add these new windows to bookeeping data structures
            self._assign_window_layers(new_commits)

            # At this point, every new commit region will only be connected
            # vertically to anything else. We now need to connect horizontally.
            self._merge_adjacent_windows()

            # Now, all windows are valid. We finish by adding buffer regions and
            # updating DAG dependencies appropriately.
            self._update_dependencies_and_dag()

        self._update_waiting_windows() # Look for dangling windows to mark as constructed

        if not new_rounds: # flush dangling windows (mark dangling windows as constructed, if no new rounds)
            self._flush_windows()

        constructed_windows = [self._get_window(window_idx) for window_idx in self.all_constructed_windows[constructed_window_count:]] # get constructed windows for all new windows

        if self.lightweight_setting > 0:
            self._clean_old_windows(self.all_constructed_windows[constructed_window_count:]) # clean old windows that are no longer used for all newly constructed windows

        self.current_round += 1
        return constructed_windows

    def _add_new_commits(self, new_commits: list[DecodingWindow]) -> None:
        """TODO"""
        new_window_start = len(self.all_windows) # get start of the new windows
        self.all_windows.extend(new_commits) # extend all windows with all the new windows
        self._unconstructed_window_indices.update({window: new_window_start+i for i,window in enumerate(new_commits)}) # update unconstructed windows with these new windows
        for i,window in enumerate(new_commits): # iterate through all new windows
            window_idx = new_window_start + i # get the curr window idx
            assert len(window.commit_region) == 1 # assert this window's commit region is only length 1 (only has 1 commit region, which is expected bc this is a newly constructed window)
            cr = window.commit_region[0] # get the commit rgn of this window
            patch, end = cr.patch, cr.round_start + cr.duration # get the patch and the end round of this commit region
            assert (patch, end) not in self.window_end_lookup # ensure that this patch, end is not already the end of some existing window
            self.window_end_lookup[(patch, end)] = (window_idx, 0) # add this window to window_end_lookup with its end round (and commit_rgn_idx=0 here)
            self.window_dag.add_node(window_idx) # add this window to the dag
            self.window_construction_wait.add(window_idx) # add this window to windows that are currently under construction

    # always defualt to adding window to the current layer, if the layer's size can tolerate it
    def _choose_layer(self, window_idx: int, possible_layers: list[int], tolerable_resulting_sizes: dict[int, int]) -> int:
        """Choose a layer to assign to a new window (which will result in it
        being merged with neighbors of the same layer) by calculating the
        expected size of the resulting merged window. 
        
        Will prefer adding to existing windows instead of inserting into a new
        layer if the resulting window is below the tolerable size for that
        layer. Assumes that first layer in list is default layer, if none are
        preferred.

        Args:
            window_idx: Index of the new window
            possible_layers: List of layer indices to consider. The first is
                treated as the default.
            tolerable_resulting_sizes: Dictionary of tolerable commit region
                sizes for each layer. If the resulting window is below this
                size, it will be added to the layer with the smallest size.
        """
        window = self._get_window(window_idx) # get window
         # get the layer size for each possible layer
        layer_sizes = {layer: 0 for layer in possible_layers}
        for idx in self._get_touching_unconstructed_window_indices(window): # get all unconstructed windows that share a boundary with this window
            for layer in possible_layers:
                if idx in self.layer_indices[layer]: # if this window idx is in layer's indices
                    layer_sizes[layer] += self._get_window(idx).commit_spacetime_volume() # then we add the spacetime volume of the current window to the layer size
                    break
        if not any(layer_sizes.values()): # if none of the layer sizes have a value (all empty)
            return possible_layers[0] # then we simply just return the first possible layer as the layer to add th window to
        # check if any layer has space to add windows to it
        elif any(0 < size < tolerable_resulting_sizes[layer] for layer,size in layer_sizes.items()): # get the layer and its current size. if size is less than the tolerable size for this layer, add the window to this layer
            # Prefer adding to existing windows if the resulting window is
            # below the tolerable size for that layer
            # first build a filtered dictionary of eligible layers (layer:size), then min picks the layer with the smallest size among the eligible ones, with default being the first possibe layer if no eligible layers exist
            return min({layer:size for layer,size in layer_sizes.items() if 0 < size < tolerable_resulting_sizes[layer]}, key=lambda k: layer_sizes[k], default=possible_layers[0])
        else: # nobody has any space, so we choose the layer whose current size is the smallest to append to it (w default=layer 0)
            return min(layer_sizes, key=lambda k: layer_sizes[k], default=possible_layers[0])

    def _assign_window_layers(self, new_commits: list[DecodingWindow]) -> None:
        """Process new commits, deciding on their window layer based on previous
        temporally-connected windows.
        """
        new_commits_to_process = new_commits.copy()
        # sort so that we process windows with history first
        def has_history(window):
            patch = window.commit_region[0].patch # get the patch of the window's commit region
            prev_window_end = window.commit_region[0].round_start # get the previous window's end (which is curr commit region's start)
            if (patch, prev_window_end) in self.window_end_lookup: # if this patch and previous end is the end of an existing window
                # we get the previous window and its commit region
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)] 
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                return not prev_commit.discard_after # if this previous commit rgn is discard after, it doesn't have subsequent history; else, it does have history
            return False # if our curr window is not the end of another window, we return False (no history)
        new_commits_to_process.sort(key=lambda window: has_history(window), reverse=True) # sort the new commits by whether they have history or not (windows with history=True are first)

        for window in new_commits_to_process: # iterate thru new windows to process
            assert len(window.commit_region) == 1 # all new windows are single commit regions
            window_idx = self._unconstructed_window_indices[window] # get the window index of this unconstructed window
            patch = window.commit_region[0].patch # get the commit region's patch of this window
            prev_window_end = window.commit_region[0].round_start # get the previous window's end (which is curr window's start)
            if (patch, prev_window_end) in self.window_end_lookup: # if this patch, prev_window_end corresponds to the end of a window that alr exists
                # get this existing previous window and its commit region
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                if prev_commit.discard_after: # means this previous window just ends after it, so not border our curr window actually bc it discards after --> so no prev window to merge with, so will be a source
                    # No previous window to merge with; this will be a source
                    self.layer_indices[self._choose_layer(window_idx, [1,0], {0:3, 1:1})].add(window_idx) # choose layer to add this windo windex to, where we prefer layers [1,0] because we're a source, and our tolerable sizes  are 3 for layer 0 and 1 for layer 1 (bc only want 1 commit region)
                elif prev_window_idx in self.layer_indices[2]: # if the previous window has layer index = 2 (meaning previous window was a sink window)
                    if len(prev_window.commit_region) < 3: # if the previous window was bordering fewer than 3 commit regions, we add our window to that window's commit region
                        # Merge with prev sink and remove from all_windows
                        assert not prev_window.constructed # not done constructing
                        self.layer_indices[2].add(window_idx) # add our window to this second layer, bc is for the sink window
                        self._merge_windows(prev_window, window) # merge windows with the previous sink window
                    else:
                        # Sink is full; this will be a source. Add prev sink
                        # as buffer region.
                        # create new source
                        self.layer_indices[1].add(window_idx)
                        # Mark prev window as constructed
                        assert not prev_window.constructed
                        self._append_to_buffers(window, prev_commit) # prev_cmt is now a buffer of this new window
                        # self.window_dag.add_edge(window_idx, prev_window_idx)
                else:
                    # This will be a sink; add buffer to prev source and
                    # mark as complete
                    assert prev_window_idx in self.layer_indices[0] or prev_window_idx in self.layer_indices[1] # previous window is a source
                    assert not prev_window.constructed # previous window not constructed
                    self.layer_indices[2].add(window_idx) # our window should act like a sink window
                    prev_window = self._append_to_buffers(prev_window, window.commit_region[0]) # append our new window to be in the buffer region of the previous window
                    # self.window_dag.add_edge(prev_window_idx, window_idx)
            else:
                # No previous window to merge with; this will be a source
                # Decide layer based on size of touching windows
                self.layer_indices[self._choose_layer(window_idx, [1,0], {0:3, 1:1})].add(window_idx)

    def _merge_adjacent_windows(self) -> None:
        # Naive approach to resolve conflicts: merge any adjacent sinks or
        # sources into each other.
        unconstructed_windows = self.get_unconstructed_windows()
        change_made = True
        while change_made:
            change_made = False
            for i, window_idx_1 in enumerate(unconstructed_windows): # iterate through all unconstructed windows
                if change_made:
                    break
                window_1 = self._get_window(window_idx_1) # get window
                for window_idx_2 in unconstructed_windows[:i]: # iterate through all windows up to this window in unconstructed windows list
                    if change_made:
                        break
                    window_2 = self._get_window(window_idx_2) # get 2nd window
                    if window_1.shares_boundary(window_2) and self._get_layer_idx(window_idx_1) == self._get_layer_idx(window_idx_2): # if the 2 windows share a boundary and are on the same layer
                        self._merge_windows(window_1, window_2) # merge the 2 windows (bc they border each other spatially)
                        unconstructed_windows.remove(window_idx_2) # remove window_idx_2 from unconostructed windows, since now it's merged with window 1
                        # unconstructed_windows = [idx - 1 if idx > window_idx_2 else idx for idx in unconstructed_windows]
                        change_made = True
                        break
                    # if no change made, we don't continue with the while loop

    # update dag dependencies appropriately, and add buffer regions
    def _update_dependencies_and_dag(self) -> None:
        unconstructed_windows = self.get_unconstructed_windows()
        for i, window_idx_1 in enumerate(unconstructed_windows):
            window_1 = self._get_window(window_idx_1)
            for window_idx_2 in unconstructed_windows[:i]:
                window_2 = self._get_window(window_idx_2)
                if window_1.shares_boundary(window_2): # if 2 window sshare boundary
                    # get layer indexes of both windows
                    layer_idx_1 = self._get_layer_idx(window_idx_1)
                    layer_idx_2 = self._get_layer_idx(window_idx_2)
                    assert not window_1.constructed and not window_2.constructed # asert both windows not constructed
                    if layer_idx_1 < layer_idx_2: # window_1 is source
                        for region in window_1.get_touching_commit_regions(window_2): # iterate thru window 2's commit regions that touch window 1
                            window_1 = self._append_to_buffers(window_1, region) # append to window 1's buffer region those commit regions that border
                        self.window_dag.add_edge(window_idx_1, window_idx_2) # add dependency btwn window 1 and 2
                    elif layer_idx_1 > layer_idx_2: # window_2 is source
                        for region in window_2.get_touching_commit_regions(window_1): # add to window 2's buffer rgns all commit rgns of window 1 that touch window 2
                            window_2 = self._append_to_buffers(window_2, region)
                        self.window_dag.add_edge(window_idx_2, window_idx_1) # add dependency btwn window 2 and 1
                    else:
                        raise ValueError("Invalid merge")

    # get the layer index of the current window index
    def _get_layer_idx(self, window_idx: int) -> int:
        layer_idx = -1
        for i, layer in enumerate(self.layer_indices):
            if window_idx in layer:
                layer_idx = i
                break
        if layer_idx == -1:
            raise ValueError("Window not in any layer")
        return layer_idx
    
    # remove window from layer and from everything else
    def _remove_window(self, window_idx: int) -> None:
        """Wrapper for super()._remove_window that updates source and sink
        indices.
        """
        layer_idx = self._get_layer_idx(window_idx)
        self.layer_indices[layer_idx].discard(window_idx)
        return super()._remove_window(window_idx)

    # merge windows wrapper -- ensures that merged windows are in the same layer
    def _merge_windows(self, window_1: DecodingWindow, window_2: DecodingWindow) -> DecodingWindow:
        """Wrapper for super()._merge_windows that updates source and sink
        indices.
        """
        window_idx_1 = self._unconstructed_window_indices[window_1]
        window_idx_2 = self._unconstructed_window_indices[window_2]

        layer_idx_1 = self._get_layer_idx(window_idx_1)
        layer_idx_2 = self._get_layer_idx(window_idx_2)
        if layer_idx_1 != layer_idx_2:
            raise ValueError("Cannot merge windows that are not assigned to same dependency layer")

        new_window = super()._merge_windows(window_1, window_2)

        return new_window

class TAlignedWindowManager(ParallelWindowManager):
    """A version of ParallelWindowManager that enforces that every window
    covering a blocking operation does not have any dependencies on future
    windows. This is to ensure that the blocking operation can be decoded
    ASAP.

    We do this by adding two new layer options to the window manager which will only
    be used for these special windows. The new layers are interleaved with the
    typical layers in ParallelWindowManager. In ParallelWindowManager, the
    layers are 0 (alternate source), 1 (main source), and 2 (sink). In this
    version, the layers are 0 (alternate source), 1 (main source), 2 (blocking
    window 1), 3 (blocking window 2), and 4 (sink). After a blocking window, we always begin a sink. This
    ensures that the blocking window never has any dependencies on future
    windows.

    """
    # ensure that blocking window is a source, so that it doesn't have any dependencies on future windows
    def __init__(self, window_builder: WindowBuilder, lightweight_setting: int = 0):
        super().__init__(window_builder, lightweight_setting=lightweight_setting)
        self.layer_indices = [set(), set(), set(), set(), set()]

    def _is_blocking_window(self, window: DecodingWindow) -> bool:
        """Check if a window is blocking another window from being decoded.
        """
        return any(instr.conditional_dependencies for instr in window.merge_instr) # check all of this window's instructions, and see if there are any later instructions that depends on this one's outcome

    def _assign_window_layers(self, new_commits: list[DecodingWindow]) -> None:
        # Process buffers in time (windows covering same patch)

        new_commits_to_process = new_commits.copy()
        # sort so that we process windows with history first
        def has_history(window):
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                return not prev_commit.discard_after # return if have this previous window (history)
            return False
        new_commits_to_process.sort(key=lambda window: has_history(window), reverse=True)

        for window in new_commits_to_process:
            assert len(window.commit_region) == 1 # all new windows are single commit regions
            window_idx = self._unconstructed_window_indices[window]
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                if prev_commit.discard_after:
                    # No previous window to merge with; this will be a source
                    if self._is_blocking_window(window):
                        self.layer_indices[self._choose_layer(window_idx, [3,2], {3:1, 2:3})].add(window_idx) # if this window is blocking, then we need to put it in layers 3 or 2 (3=main, 2=backup)
                    else:
                        self.layer_indices[self._choose_layer(window_idx, [1,0], {1:1, 0:3})].add(window_idx) # if window not blocking, put in layers [1,0]
                elif prev_window_idx in self.layer_indices[4]: # if prev window is a sink
                    if self._is_blocking_window(window): # if I'm a blocking window and prev window is a sink, then I should be added as a blocking source, and prev window is one of my buffer regions 
                        self.layer_indices[3].add(window_idx)
                        assert not prev_window.constructed
                        self._append_to_buffers(window, prev_commit)
                        # self.window_dag.add_edge(window_idx, prev_window_idx)
                    # need this bc sink is not full, and sinks must accummulate 3 commit regions to be a single decodable unit
                    elif len(prev_window.commit_region) < 3: # if prev window is a sink and it's commit region is less than 3, then I'm one of the source windows of the prev window, add to layer 4 bc I merge w prev window
                        # Merge with prev sink and remove from all_windows
                        assert not prev_window.constructed
                        self.layer_indices[4].add(window_idx)
                        self._merge_windows(prev_window, window)
                    else:
                        # Sink is full; this will be a source. Add prev sink
                        # as buffer region.
                        # create new source
                        self.layer_indices[1].add(window_idx) # bc we know it is non blocking here
                        # Mark prev window as constructed
                        assert not prev_window.constructed
                        self._append_to_buffers(window, prev_commit) # prev sink appended as buffer region
                        self.window_dag.add_edge(window_idx, prev_window_idx)
                else:
                    # Previous is a source; this will be a sink
                    assert not prev_window.constructed
                    if self._is_blocking_window(window):
                        assert self._get_layer_idx(prev_window_idx) in [0,1]
                        self.layer_indices[3].add(window_idx) # I am a vlocking operation
                        prev_window = self._append_to_buffers(prev_window, window.commit_region[0]) # I am a sink of the previous window
                        # self.window_dag.add_edge(prev_window_idx, window_idx)
                    else: # previous window must not be a sink
                        assert self._get_layer_idx(prev_window_idx) in [0,1,2,3]
                        self.layer_indices[4].add(window_idx) # add me to the sink list
                        prev_window = self._append_to_buffers(prev_window, window.commit_region[0]) # I am now a buffer region of the previous window bc I'm a sink
                        # self.window_dag.add_edge(prev_window_idx, window_idx)
            else:
                # No previous window to merge with; this will be a source
                if self._is_blocking_window(window):
                    self.layer_indices[self._choose_layer(window_idx, [3,2], {3:1, 2:3})].add(window_idx) # blocking window source
                else:
                    self.layer_indices[self._choose_layer(window_idx, [1,0], {1:1, 0:3})].add(window_idx) # non blocking window source