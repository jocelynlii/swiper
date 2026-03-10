from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
import random
from collections import defaultdict

from swiper.lattice_surgery_schedule import Instruction
from swiper.device_manager import SyndromeRound

@dataclass(frozen=True)
class SpacetimeRegion:
    """A region of spacetime in the decoding volume.

    Attributes:
        patch: Spatial coordinates of region.
        round_start: Measurement round starting the region.
        duration: Temporal length, in units of measurement rounds.
        num_spatial_boundaries: Number of spatial faces which are not shared
            with another region.
        initialized_patch: If True, this region was the first of the patch.
        discard_after: If True, this patch was discarded after this region.
        prior_t: If True, a T gate was injected in the round before this region.
        merge_instr: MERGE instruction if applicable
    """
    patch: tuple[int, int]
    round_start: int
    duration: int
    num_spatial_boundaries: int # this means the faces that don't border another patch
    initialized_patch: bool = False
    discard_after: bool = False # this is essentially the last "region" of the patch
    prior_t: bool = False
    merge_instr: Instruction | None = None

    # check if round is contained by checking if it operates on the same patch as this region and temporally it is within this patch's duration, starting from round start
    def contains_syndrome_round(self, *, patch: tuple[int, int] | None = None, round: int | None = None, syndrome_round: SyndromeRound | None = None) -> bool:
        """Check if a syndrome round is contained in this region. Either patch
        and round must be given, or a syndrome round object.
        """
        if patch is not None and round is not None: # checks if the patch and round given are within the syndrome round's patch and duration (starting from round start)
            return patch == self.patch and self.round_start <= round < self.round_start + self.duration
        elif syndrome_round is not None: # check if the current syndrome round is operating on the same patch, and is within this patch's duration (starting from round start)
            return syndrome_round.patch == self.patch and self.round_start <= syndrome_round.round < self.round_start + self.duration
        else:
            raise ValueError('Either patch and round or a syndrome round must be given.')
    
    def shares_timelike_boundary(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region shares a timelike boundary with another region.
        """
        return (
            self.patch == other.patch
            and
            (
                ( # make sure that this other region is not the last region, and that this other region's end round border corresponds with the current regions' start round border
                    (not other.discard_after)
                    and
                    self.round_start == other.round_start + other.duration
                )
                or # below is just the flipped verison of the above
                ( # make sure that this region is not the last region, but that this region's end round border corresponds with the other region's start round border
                    (not self.discard_after)
                    and
                    other.round_start == self.round_start + self.duration
                )
            )
        )
    
    def shares_spacelike_boundary(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region shares a spacelike boundary with another region.
        """
        # make sure that these instructions are both merge instructions, and they are part of the same merge inst (only way they share a boundary), 
        # and the tuple of (self.patch, other.patch) or vice versa is in the merge_inst's merge faces (shared boundaires btwn merged patches)
        return (
            self.merge_instr is not None and other.merge_instr is not None
            and self.merge_instr == other.merge_instr
            and ((self.patch, other.patch) in self.merge_instr.merge_faces or (other.patch, self.patch) in self.merge_instr.merge_faces)
        )
    
    # check if shares space or timelike boundary
    def shares_boundary(self, other: 'SpacetimeRegion') -> bool:
        return self.shares_timelike_boundary(other) or self.shares_spacelike_boundary(other)
    
    # check if the 2 regions overlap (draw diagram to see that these conditions do indeed ensure this)
    def overlaps(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region overlaps with another region.
        """
        return self.patch == other.patch and self.round_start < other.round_start + other.duration and other.round_start < self.round_start + self.duration
    
    def shares_edge(self, other: 'SpacetimeRegion') -> bool:
        return (
            (   # spacelike edge
                np.linalg.norm(np.array(self.patch) - np.array(other.patch)) == 1
                and (
                    self.round_start == other.round_start + other.duration
                    or other.round_start == self.round_start + self.duration
                )
            )
            or (# timelike edge
                np.abs(np.array(self.patch) - np.array(other.patch)).sum() == 2
                and other.round_start <= self.round_start < other.round_start + other.duration
            )
        )

    # return a string of all info about this spacetime region
    def __repr__(self):
        return f'Region({self.patch}, {self.round_start}, {self.duration}, {self.num_spatial_boundaries}, {self.initialized_patch}, {self.discard_after})'

@dataclass(frozen=True)
class DecodingWindow:
    """A decoding window with commit and buffer regions.

    Attributes:
        commit_region: Spacetime region that is commited after decoding.
        buffer_regions: Spacetime regions that are not commited after decoding.
                        The boundary between a buffer region and the commit region
                        forms a decoding dependency from this window to
                        adjacent windows.
        merge_instr: MERGE instruction for spatial buffers if necessary.
        parent_instr_idx: List of indices of instructions that generated this window.
        constructed: True if window is finished being constructed with buffers 
    """
    commit_region: tuple[SpacetimeRegion, ...] # 
    buffer_regions: frozenset[SpacetimeRegion] # immutable set (set is where all objects are unique)
    merge_instr: frozenset[Instruction]
    parent_instr_idx: frozenset[int] # list of insts that generated this window --> this is how we map a window to the instructions (insts here are lattice surgery operations)
    window_idx: int # this is a sequentially increasing counter for the number of window we're on (counter increments every time we generate a new window)
    constructed: bool
    speculation_accuracy: float # ADDED -- speculation accuracy in window level
    speculation_time: int # ADDED -- speculation time on window level
    decoding_time_fn: Callable[[int], int] # ADDED -- decoding time function on window level
    decoding_time_fn_str: str

    # this spacetime volume is in units of measurement rounds, since duration is the temporal length in terms of measurement rounds
    # volume is in spacetime, meaning we have to calculate the volume in terms of both temporal rounds and d^2 (d^2 accommodates for space)
    def commit_spacetime_volume(self) -> int:
        """Calculate the spacetime volume of the commit region, in units of
        rounds*d^2."""
        return sum(region.duration for region in self.commit_region)
    
    def buffer_spacetime_volume(self) -> int:
        """Calculate the spacetime volume of the buffer regions, in units of
        rounds*d^2."""
        return sum(region.duration for region in self.buffer_regions)

    def total_spacetime_volume(self) -> int:
        """Calculate the total spacetime volume of this window, in units of
        rounds*d^2."""
        return self.commit_spacetime_volume() + self.buffer_spacetime_volume()

    def shared_timelike_boundaries(self, other: 'DecodingWindow') -> list[tuple[SpacetimeRegion, SpacetimeRegion]]:
        shared_boundaries = []
        # iterate throguh all spacetime regions in commit and buffer, check if these regions share a timelike boundary, and if so, append to shared_boundaries the 2 regions
        # this is checking if the 2 regions share a TEMPORAL boundary (but for this to be the case, they do have to be "operating" on the same patch)
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_timelike_boundary(other_region):
                    shared_boundaries.append((region, other_region))
        return shared_boundaries

    # same as above function, but returns back a boolean which is True if self and other have any region that shares a timelike (temporal) boundary, else it returns False
    def shares_timelike_boundary(self, other: 'DecodingWindow') -> bool:
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_timelike_boundary(other_region):
                    return True
        return False

    # check if any region in self or other's commit region shares a spacelike boundary -- if so, return a list of these regions (spacelike boundary is shared space boundaries btwn merged patches)
    def shared_spacelike_boundaries(self, other: 'DecodingWindow') -> list[tuple[SpacetimeRegion, SpacetimeRegion]]:
        shared_boundaries = []
        # iterate thru all regions of self and other
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_spacelike_boundary(other_region):
                    shared_boundaries.append((region, other_region))
        return shared_boundaries
    
    # same as above function, but returns true if any region within self or other share a spacelike boundary; else, returns false
    def shares_spacelike_boundary(self, other: 'DecodingWindow') -> bool:
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_spacelike_boundary(other_region):
                    return True
        return False

    # check if shares either spacelike or timelike boundary btwn self and other
    def shares_boundary(self, other: 'DecodingWindow') -> bool:
        return self.shares_timelike_boundary(other) or self.shares_spacelike_boundary(other)

    # get commit regions of other that share a spacelike boundary with self
    def get_touching_commit_regions(self, other: 'DecodingWindow') -> list[SpacetimeRegion]:
        """Get the commit regions of `other` that are touching (share a
        boundary)."""
        shared_boundaries = self.shared_spacelike_boundaries(other)
        adjacent_regions = []
        for region in self.commit_region:
            for other_region in other.commit_region:
                if (region, other_region) in shared_boundaries or (other_region, region) in shared_boundaries:
                    adjacent_regions.append(other_region) # we want to append the commit regions of other, not self
        return adjacent_regions
    
    # count the total amount of shared boundaries (both time and spacelike boundaries)
    def count_touching_faces(self, other: 'DecodingWindow') -> int:
        return len(self.shared_timelike_boundaries(other)) + len(self.shared_spacelike_boundaries(other))

    # if buffer regions overlap with commit regions, return true; else, return false (this overlap check is moreso in a temporal sense)
    def overlaps(self, other: 'DecodingWindow') -> bool:
        """Check if buffer regions of this window overlap with commit regions of
        other, or vice versa."""
        for self_commit in self.commit_region:
            for other_buffer in other.buffer_regions:
                if self_commit.overlaps(other_buffer):
                    return True
        for other_commit in other.commit_region:
            for self_buffer in self.buffer_regions:
                if other_commit.overlaps(self_buffer):
                    return True
        return False

    # for each buffer region, append the commit regions that share a boundary with it (key=buffer rgn, val=list of commit rgns touching it)
    def buffer_boundary_commits(self) -> dict[SpacetimeRegion, list[SpacetimeRegion]]:
        """Returns dict mapping each buffer region to all the commit regions
        touching it."""
        commits = {}
        for br in self.buffer_regions:
            for cr in self.commit_region:
                if br.shares_boundary(cr):
                    commits.setdefault(br, []).append(cr)
        return commits
    
    # return string of all the info of this DecodingWindow
    def __repr__(self):
        return f'Window({self.commit_region}, {self.buffer_regions}, {self.parent_instr_idx}, {self.constructed}, EDIT: {self.speculation_accuracy}, {self.speculation_time}, {self.decoding_time_fn})'

class WindowBuilder():
    def __init__(self, d: int, lightweight_setting: int = 0, decoder_parameters: list[dict[str, float]] = None, window_parameters: dict | None = None, schedule_insts: list | None = None) -> None:
        self._patch_groups: dict[tuple[int, int], list[int]] = {}
        self._all_rounds: list[SyndromeRound] = []
        self._waiting_rounds: set[int] = set()
        self._inject_t_rounds: set[int] = set()
        self._inject_t_rounds_dict: dict[tuple[int, int], list[int]] = dict()
        self._total_rounds_processed: int = 0
        self._created_window_count: int = 0
        self.d: int = d
        self.lightweight_setting = lightweight_setting
        self.decoder_paramters = decoder_parameters
        self.window_parameters = window_parameters
        self.schedule_insts = schedule_insts
        self.flip_flag= False

    # I don't think I need to do this for cmt rgns because I've only ever seen one SR for commit region
    # But I do think I need to do this for buffer rgns because I've seen many SRs for buffer region
    # def build_index(self, regions: tuple[SpacetimeRegion, ...]): 
    #     index = defaultdict(list)
    #     for r in regions:
    #         index[(r.patch, r.duration, r.num_spatial_boundaries, r.initialized_patch, r.discard_after)].append(r)
    #     return index

    # this is exceedingly inefficient...
    # maybe just search +/- a few windows, or else this is going to blow up with how slow it is
    # returns the dict of the matching window in the config file (the 'window_info' dict)
    # def window_in_config(self, window: DecodingWindow):
    #     if self.window_parameters is not None:
    #         same_inst_windows = {}
    #         same_inst_and_commit_windows = {}

    #         # search through all windows in our config file
    #         # first narrow down search space by parent inst index
    #         # create a list of all window dicts (window_info dicts) that have 
    #         for idx, val in self.window_parameters.items(): # curr_window is a dict
    #             curr_window = val['window_info'] # get the window info of the current window we're looking at
    #             if window.parent_instr_idx == frozenset(curr_window['parent_instr_idx']):
    #                 # same_inst_windows.append({idx: curr_window})
    #                 same_inst_windows[idx] = curr_window

    #         # then from this narrowed down list of windows (contains window_info dicts), 
    #         # search these windows for windows with matching commit regions
    #         # create a list of all window dicts (window_info dicts) that have the same commit region (in addition to already same parent inst)
    #         for idx, curr_window in same_inst_windows.items():
    #             if len(curr_window['commit_region']) > 1 or len(window.commit_region) > 1: # we should only have one commit region
    #                 raise ValueError(f"Commit Region Not Length 1: {len(curr_window['commit_region'])} {len(window.commit_region)}")
                
    #             if (tuple(curr_window['commit_region'][0]['patch']) == window.commit_region[0].patch 
    #                 and curr_window['commit_region'][0]['duration'] == window.commit_region[0].duration
    #                 and curr_window['commit_region'][0]['num_spatial_boundaries'] == window.commit_region[0].num_spatial_boundaries
    #                 and curr_window['commit_region'][0]['initialized_patch'] == window.commit_region[0].initialized_patch
    #                 and curr_window['commit_region'][0]['discard_after'] == window.commit_region[0].discard_after):
    #                 same_inst_and_commit_windows[idx] = curr_window

    #         # now from same commit rgn and parent insts windows, we want to search for if they have the same buffer region
    #         # because buffer region can have multiple spacetime regions, need to compare their frozensets (a bit more hard)
    #         for idx, curr_window in same_inst_and_commit_windows.items():
    #             window_obj_regions = {(o.patch, o.duration, o.num_spatial_boundaries, o.initialized_patch, o.discard_after) for o in window.buffer_regions}
    #             window_dict_regions = {(tuple(d['patch']), d['duration'], d['num_spatial_boundaries'], d['initialized_patch'], d['discard_after']) for d in curr_window['buffer_regions']}
    #             if window_obj_regions == window_dict_regions:
    #                 print("MATCH FOUND")
    #                 # remove this match from the window parameters dict
    #                 del self.window_parameters[idx]
    #                 return curr_window
                
    #     return None # either no window parameters, or didn't find a match for the current window in window parameters

    # new_rounds should all be in the same round (just diff regions)
    def build_windows(
            self, 
            new_rounds: list[SyndromeRound],
        ) -> list[DecodingWindow]:
        """Process new rounds and output any windows with complete commit
        regions.
        
        Args:
            new_rounds: List of new syndrome rounds to process. Should all be
                from the same cycle of the device.

        Returns:
            List of newly-completed decoding windows.
        """
        if not new_rounds or len(new_rounds) == 0: # if no new rounds, we set curr_round to the setting that allows us to work on windows in the backlog
            # Time to chug through that backlog
            curr_round = -1
        else: # append new rounds to global all rounds array; update waiting rounds, remember where it started, add to either patch groups for all the round's patches, or inject_t_rounds depending on the round's instruction
            curr_round = new_rounds[0].round # get the round number that we're on for the first thing in new_Rounds
            assert all([round.round == curr_round for round in new_rounds]) # ensure that all the rounds in new_rounds are in the same round (so they're just in a different region)

            new_round_start = len(self._all_rounds) # get the index of where the new rounds begin
            self._all_rounds.extend(new_rounds) # add new rounds to all rounds
            # now all of these new rounds are considered as waiting rounds, except for rounds with the INJECT_T instruction name (only non inject_t because inject_t don't need to be decoded)
            self._waiting_rounds.update([i+new_round_start
                                        for i,round in enumerate(new_rounds)
                                        if round.instruction.name != 'INJECT_T']) # T injection is not decoded 
            # if round instruction name is not INJECT_T, then we add this patch to patch groups, and add this inst's index to it
            for i,round in enumerate(new_rounds):
                if round.instruction.name != 'INJECT_T':
                    self._patch_groups.setdefault(round.patch, []).append(i+new_round_start)
                else: # add to a different inject_t dict if this inst is an inject_t op
                    self._inject_t_rounds_dict.setdefault(round.patch, []).append(i+new_round_start) # use this dict to label other instructions' prior_t

        new_windows = []

        # get the rounds corresponding to a particular patch
        # attempt to form a commit region per patch
        for patch, round_indices in list(self._patch_groups.items()):
            rounds = [self._all_rounds[round_idx] for round_idx in round_indices] # gather all (pending) rounds for this patch
            min_round = min(rounds, key=lambda x: x.round) # get the minimum round number
            max_round = max(rounds, key=lambda x: x.round) # get the maximum round number
            duration = self.d # get the duration
            # decide how long the commit region should be
            if max_round.round != curr_round or max_round.discard_after: # if the maximum round number is not the current round number or we discard right after the max round
                # Dangling rounds (e.g. S gate cap)
                # aka: commit region ends right now, since the most recent round for this patch is older than the device's current round (no new round for it this cycle)
                # if nothing for this patch arrived this cycle, or we're discarding right after, this means the commit region should end right here (cut a window ending at the latest available round)
                duration = max_round.round - min_round.round + 1
            elif min_round.instruction.name != 'MERGE' and max_round.instruction.name == 'MERGE':
                # Aligning windows with merges is non-negotiable due to the need for spatial buffers
                # cut the commit region to before the merge region begins, since we want the non-merge region to be 1 window, and the merge region to be a 2nd separate window
                # (start non-merge, latest merge)
                junk_round_end = max(rounds, key=lambda x: x.round * (0 if x.instruction.name == 'MERGE' else 1)) # max round of a non-merge instruction
                round_indices = [round_idx for round_idx in round_indices if self._all_rounds[round_idx].round <= junk_round_end.round] # get all round indices whose rounds are less than the junk_round_end
                rounds = [self._all_rounds[round_idx] for round_idx in round_indices] # get all rounds of these round indices
                duration = junk_round_end.round - min_round.round + 1 # duration is the end junk round minus the min round
            elif min_round.instruction.name == 'MERGE' and max_round.instruction.name != 'MERGE': # flipped from above elif clause
                # if we start in a merge rgn then go to a non-merge rgn, we want the merge rgn to be 1 window, and the non-merge rgn to be a 2nd separate window
                # (start merge, latest non-merge)
                junk_round_end = max(rounds, key=lambda x: x.round * (1 if x.instruction.name == 'MERGE' else 0)) # get the max round of a merge inst
                round_indices = [round_idx for round_idx in round_indices if self._all_rounds[round_idx].round <= junk_round_end.round] # get all round indices whose rounds are are less than the junk round end
                rounds = [self._all_rounds[round_idx] for round_idx in round_indices] # get all the rounds for thes round_indices
                duration = junk_round_end.round - min_round.round + 1 # get the duration
            elif (max_round.round - min_round.round) + 1 < duration:
                # Not enough rounds to create a window
                # because it's less than the duration
                # we need the commit region to be at least duration length, so if it's not, we can't create a commit region with it
                # print("IN THIS BREAK3?")
                continue
            # compute commit region metadata
            max_round = max(rounds, key=lambda x: x.round) # get the max round in the rounds
            # compute "exposed" faces, which don't count faces that are shared from a merge
            num_spatial_boundaries = 4 - sum(patch in face for face in min_round.instruction.merge_faces) # get the number of spatial boundaries (no boundaries at merge faces bc these are "shared"/"merged" boundaries)
            prior_t = False # if patch not initialized, and have INJECT_T exactly at round right before min round, prior_t=True
            # only enter if statement if this patch is not newly initialized (bc if so, won't have a "prior") and only proceed if this patch has ever had any inject_t round recorded
            if not min_round.initialized_patch and patch in self._inject_t_rounds_dict: # if it's not the initialized patch and patch is an inject_t patch
                for idx in reversed(self._inject_t_rounds_dict[patch]): # go through all different inject_t rounds for this patch (the diff inject_t rounds occur at different times temporally)
                    t_round = self._all_rounds[idx]
                    if t_round.round == min_round.round-1: # if the t round is the round right before the start of our current window, then prior_t is true
                        # print("IN THIS BREAK?")
                        prior_t = True
                        break
                    elif t_round.round < min_round.round-1: # if the t round is less than being before the start of our current window, we break, becuase the other rounds we iterate thru in the for loop are only going to be smaller
                        # print("IN THIS BREAK2?")
                        break
            # this contains the lattice surgery instructions that contributed rounds to this window
            parent_instr_idx = frozenset([round.instruction_idx for round in rounds]) # get a set of all round's instruction indexes
            commit_region = SpacetimeRegion( # create the commit region
                patch=patch,
                round_start=min_round.round,
                duration=duration,
                num_spatial_boundaries=num_spatial_boundaries,
                initialized_patch=min_round.initialized_patch,
                discard_after=max_round.discard_after,
                prior_t=prior_t,
                merge_instr=min_round.instruction if min_round.instruction.name == 'MERGE' else None, # if is merge start, set this
            )
            # curr_speculation_accuracy = 0
            # if self.window_parameters is not None:
            #     self.window_in_config()
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_speculation_accuracy = self.window_parameters[str(self._created_window_count)]['speculation_accuracy']
            #     else: # just do default
            #         curr_speculation_accuracy = 0.6
            # else:
            #     curr_speculation_accuracy = 0.6 # manually change to default

            # curr_speculation_time = 0
            # if self.window_parameters is not None:
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_speculation_time = self.window_parameters[str(self._created_window_count)]['speculation_time']
            #     else: # just do default otherwise
            #         curr_speculation_time = 1
            # else:
            #     curr_speculation_time = 1 # manually change to default

            # curr_decoding_time_fn = None # lambda _: 14 # None 
            # if self.window_parameters is not None:
            #     # print("curr decoding time fn ", self.window_parameters[str(self._created_window_count)]['decoding_time_fn'])
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_decoding_time_fn = eval(self.window_parameters[str(self._created_window_count)]['decoding_time_fn'])
            #     else: # do default otherwise
            #         curr_decoding_time_fn = lambda _: 7
            # else:
            #     curr_decoding_time_fn = lambda _: 7

            new_windows.append(DecodingWindow(
                commit_region=(commit_region,),
                buffer_regions=frozenset(), # no buffer regions so far
                merge_instr=frozenset() if min_round.instruction.name != 'MERGE' else frozenset([min_round.instruction]), # if is merge start, set this
                parent_instr_idx=parent_instr_idx,
                window_idx=self._created_window_count,
                constructed=False,
                speculation_accuracy= random.uniform(0, 1), # 0.8 if (sorted(parent_instr_idx)[0] == -1 or not self.schedule_insts[sorted(parent_instr_idx)[0]].instruction.t_gate_bool ) else 0.5, # or self.schedule_insts[sorted(parent_instr_idx)[0]].instruction.name == "COND_S"  # random.uniform(0, 1), # 0.9, # curr_speculation_accuracy, # 0.6 # np.random.rand()
                speculation_time=1, # curr_speculation_time,
                decoding_time_fn= lambda _: 14,
                decoding_time_fn_str="lambda _: 14"
            ))
            # print("build windows ", parent_instr_idx, commit_region, rounds)
            self._created_window_count += 1

            # removed consumed round indices for this patch from patch_groups
            self._patch_groups[patch] = [round_idx for round_idx in self._patch_groups[patch] if round_idx not in round_indices] 
            if not self._patch_groups[patch]:
                self._patch_groups.pop(patch)
            self._waiting_rounds -= set(round_indices) # remove same indices from waiting rounds too # these round indices are now not waiting since they're part of a window
            if self.lightweight_setting > 0:
                for round_idx in round_indices:
                    self._all_rounds[round_idx] = None # update these used round indices to None to save memory

        self._total_rounds_processed += len(new_rounds)

        return new_windows # return new windows created in this call

    def flush(self):
        """Flush all remaining rounds into windows, allowing smaller windows
        than the usual size.
        """
        new_windows = []
        for patch, round_indices in list(self._patch_groups.items()): # iterate through all patches
            rounds = [self._all_rounds[round_idx] for round_idx in round_indices] # get all rounds per patch
            min_round = min(rounds, key=lambda x: x.round) # get min and max round for all rounds in the patch
            max_round = max(rounds, key=lambda x: x.round)
            duration = max_round.round - min_round.round + 1 # get the duration for the patch
            num_spatial_boundaries = 4 - sum(patch in face for face in min_round.instruction.merge_faces) # get "exposed" boundaries, which aren't part of merged faces
            parent_instr_idx = frozenset([round.instruction_idx for round in rounds]) # get all the instruction indexes that led to these rounds for the patch
            # create a commit region with the current min_round (no duration constraint like in the previous build_windows function)
            commit_region = SpacetimeRegion(
                patch=patch,
                round_start=min_round.round,
                duration=duration,
                num_spatial_boundaries=num_spatial_boundaries,
                initialized_patch=min_round.initialized_patch, # get first ronud's initialized_patch
                discard_after=True,
                merge_instr=min_round.instruction if min_round.instruction.name == 'MERGE' else None,
            )

            # curr_speculation_accuracy = 0
            # if self.window_parameters is not None:
            #     # print('window params in window builder', self.window_parameters)
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_speculation_accuracy = self.window_parameters[str(self._created_window_count)]['speculation_accuracy']
            #     else: # just do default
            #         curr_speculation_accuracy = 0.6
            # else:
            #     curr_speculation_accuracy = 0.6 # manually change to default

            # curr_speculation_time = 0
            # if self.window_parameters is not None:
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_speculation_time = self.window_parameters[str(self._created_window_count)]['speculation_time']
            #     else: # just do default otherwise
            #         curr_speculation_time = 1
            # else:
            #     curr_speculation_time = 1 # manually change to default

            # curr_decoding_time_fn = None # lambda _: 14 # None 
            # if self.window_parameters is not None:
            #     # print("curr decoding time fn ", self.window_parameters[str(self._created_window_count)]['decoding_time_fn'])
            #     if str(self._created_window_count) in self.window_parameters:
            #         curr_decoding_time_fn = eval(self.window_parameters[str(self._created_window_count)]['decoding_time_fn'])
            #     else: # do default otherwise
            #         curr_decoding_time_fn = lambda _: 7
            # else:
            #     curr_decoding_time_fn = lambda _: 7

            # create a decoding window with this commit region
            new_windows.append(DecodingWindow(
                commit_region=(commit_region,),
                buffer_regions=frozenset(),
                merge_instr=frozenset() if min_round.instruction.name != 'MERGE' else frozenset([min_round.instruction]), # depends on whether min_round is a merge instruction
                parent_instr_idx=parent_instr_idx,
                window_idx=self._created_window_count,
                constructed=False,
                speculation_accuracy= random.uniform(0, 1), # 0.8 if (sorted(parent_instr_idx)[0] == -1 or not self.schedule_insts[sorted(parent_instr_idx)[0]].instruction.t_gate_bool) else 0.5, #  or self.schedule_insts[sorted(parent_instr_idx)[0]].instruction.name == "COND_S" # random.uniform(0, 1), # 0.9, # curr_speculation_accuracy, # 0.6 # np.random.rand()
                speculation_time= 1, # curr_speculation_time,
                decoding_time_fn= lambda _: 14,
                decoding_time_fn_str="lambda _: 14"
            ))
            self._created_window_count += 1
            self._patch_groups.pop(patch) # clear this patch's pending rounds from patch groups and waiting rounds
            self._waiting_rounds -= set(round_indices)

        assert len(self._waiting_rounds) == 0 # should be done procesing all waiting rounds by this point
        return new_windows
    
    def get_incomplete_instructions(self): # return all waiting rounds
        return {self._all_rounds[round].instruction_idx for round in self._waiting_rounds}