import stim
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from swiper.lattice_surgery_schedule import Instruction

# ----------------------------
# Toy schedule IR
# ----------------------------
@dataclass
class Instr:
    op: str                      # "IDLE", "INJECT_T", "MERGE", "DISCARD", "Y_MEAS", ...
    patch: Optional[str] = None   # e.g. "A"
    patch2: Optional[str] = None  # e.g. "B" for MERGE(A,B)
    rounds: int = 1               # for IDLE, MERGE, etc.

# ----------------------------
# Toy patch -> qubits mapping
# Replace with your real geometry mapping.
# ----------------------------
class PatchMap:
    def __init__(self):
        self.patch_qubits: Dict[str, List[int]] = {}
        self.next_q = 0

    def alloc_patch(self, name: str, n_data: int) -> None:
        qs = list(range(self.next_q, self.next_q + n_data))
        self.next_q += n_data
        self.patch_qubits[name] = qs

    def qubits(self, name: str) -> List[int]:
        return self.patch_qubits[name]

# ----------------------------
# Translator
# ----------------------------
class LatticeSurgeryToStim:
    def __init__(self, patch_map: PatchMap):
        self.pm = patch_map
        self.c = stim.Circuit()

        # We'll store the last measurement record index for each patch's Y_MEAS, for conditionals.
        self.last_y_meas_rec: Dict[str, int] = {}

    def append_idle(self, patch: str, rounds: int = 1):
        # In a real model, "idle" would include stabilizer rounds.
        # As a placeholder, we just insert TICKs.
        for _ in range(rounds):
            self.c.append("TICK")

    def append_inject_t_stub(self, patch: str):
        # IMPORTANT: This is a STUB.
        # A common placeholder: prepare |+> on data qubits (RX) and optionally inject some noise.
        qs = self.pm.qubits(patch)
        self.c.append("R", qs)      # reset to |0>
        self.c.append("H", qs)      # make |+>
        # Optional: treat injection imperfection as Pauli noise on the patch
        # self.c.append("X_ERROR", qs, 0.001)  # example
        self.c.append("TICK")

        # If you *really* want an "S instead of T" placeholder:
        # self.c.append("S", qs)
        # but this is just to keep everything Clifford.

    def append_merge_stub(self, patch_a: str, patch_b: str, rounds: int = 1):
        # Real MERGE is a parity measurement across a boundary with ancillas over multiple rounds.
        # Placeholder: entangle representative qubits then measure an ancilla parity.
        qa = self.pm.qubits(patch_a)[0]
        qb = self.pm.qubits(patch_b)[0]
        anc = self.pm.next_q
        self.pm.next_q += 1

        # Simple parity gadget (Z⊗Z parity as an example)
        self.c.append("R", [anc])
        for _ in range(rounds):
            self.c.append("CZ", [qa, anc])
            self.c.append("CZ", [qb, anc])
            self.c.append("TICK")
        self.c.append("M", [anc])  # parity result in measurement record
        self.c.append("TICK")

    def append_discard(self, patch: str):
        # Discard = measure and forget, or deallocate.
        # Placeholder: measure all data qubits in Z and do nothing with the results.
        qs = self.pm.qubits(patch)
        self.c.append("M", qs)
        self.c.append("TICK")

    def append_y_meas_stub(self, patch: str):
        # Placeholder for "logical Y measurement of a patch":
        # measure all data qubits in Y and parity them in software (not modeled).
        # We'll just measure one representative qubit in Y for wiring conditionals.
        q = self.pm.qubits(patch)[0]
        self.c.append("MY", [q])
        # Record index: stim uses rec[-1] to refer to last measurement.
        # We'll store the absolute index via counting is annoying; easiest is store "rec[-1]" usage.
        self.last_y_meas_rec[patch] = -1
        self.c.append("TICK")

    def append_cond_s_from_y(self, patch_target: str, meas_patch: str):
        # This models: if Y measurement == 1, apply S (or S_DAG) to target patch.
        # Stim supports classical control via "CX rec[...] q" style and also via blocks.
        #
        # We'll use a loop-free conditional:
        #   - Convert measurement record into a Z correction using feedback gadgets is tricky;
        #   - But for a starter, use "CORRELATED_ERROR" isn't conditional.
        #
        # Best simple way: use stim's classical control on Pauli gates:
        #   e.g., "CZ rec[-1] q" is allowed for some ops? Actually Stim supports
        #   measurement-controlled Pauli ops like "CX rec[-1] q", "CZ rec[-1] q".
        #   There isn't a direct controlled-S.
        #
        # So for a *starter*:
        # - Track the conditional S in your own classical side data structure (Pauli frame / Clifford frame),
        #   OR
        # - Approximate conditional S by updating a frame instead of applying it in-circuit.
        #
        # We'll do the recommended approach: frame tracking hook.
        pass

    def translate(self, schedule: List[Instr]) -> stim.Circuit:
        for ins in schedule:
            if ins.op == "IDLE":
                self.append_idle(ins.patch, ins.rounds)
            elif ins.op == "INJECT_T":
                self.append_inject_t_stub(ins.patch)
            elif ins.op == "MERGE":
                self.append_merge_stub(ins.patch, ins.patch2, ins.rounds)
            elif ins.op == "Y_MEAS":
                self.append_y_meas_stub(ins.patch)
            elif ins.op == "DISCARD":
                self.append_discard(ins.patch)
            else:
                raise ValueError(f"Unknown op {ins.op}")
        return self.c

# ----------------------------
# Example usage
# ----------------------------
pm = PatchMap()
pm.alloc_patch("A", n_data=9)  # toy "patch A"
pm.alloc_patch("B", n_data=9)  # toy "patch B" (neighbor)

sched = [
    Instr("IDLE", patch="A", rounds=1),
    Instr("INJECT_T", patch="B"),
    Instr("MERGE", patch="A", patch2="B", rounds=1),
    Instr("DISCARD", patch="B"),
    # COND_S decomposed:
    Instr("MERGE", patch="A", patch2="B", rounds=1),
    Instr("Y_MEAS", patch="B"),
    Instr("IDLE", patch="A", rounds=1),
    Instr("DISCARD", patch="B"),
]

translator = LatticeSurgeryToStim(pm)
circuit = translator.translate(sched)
print(circuit)


class LatticeSurgeryToStim:
    def __init__(self, patch_map: PatchMap):
        self.pm = patch_map
        self.c = stim.Circuit()

        # We'll store the last measurement record index for each patch's Y_MEAS, for conditionals.
        self.last_y_meas_rec: Dict[str, int] = {}

    def append_idle(self, patch: str, rounds: int = 1):
        # In a real model, "idle" would include stabilizer rounds.
        # As a placeholder, we just insert TICKs.
        for _ in range(rounds):
            self.c.append("TICK")

    def append_inject_t_stub(self, patch: str):
        # IMPORTANT: This is a STUB.
        # A common placeholder: prepare |+> on data qubits (RX) and optionally inject some noise.
        qs = self.pm.qubits(patch)
        self.c.append("R", qs)      # reset to |0>
        self.c.append("H", qs)      # make |+>
        # Optional: treat injection imperfection as Pauli noise on the patch
        # self.c.append("X_ERROR", qs, 0.001)  # example
        self.c.append("TICK")

        # If you *really* want an "S instead of T" placeholder:
        # self.c.append("S", qs)
        # but this is just to keep everything Clifford.

    def append_merge_stub(self, patch_a: str, patch_b: str, rounds: int = 1):
        # Real MERGE is a parity measurement across a boundary with ancillas over multiple rounds.
        # Placeholder: entangle representative qubits then measure an ancilla parity.
        qa = self.pm.qubits(patch_a)[0]
        qb = self.pm.qubits(patch_b)[0]
        anc = self.pm.next_q
        self.pm.next_q += 1

        # Simple parity gadget (Z⊗Z parity as an example)
        self.c.append("R", [anc])
        for _ in range(rounds):
            self.c.append("CZ", [qa, anc])
            self.c.append("CZ", [qb, anc])
            self.c.append("TICK")
        self.c.append("M", [anc])  # parity result in measurement record
        self.c.append("TICK")

    def append_discard(self, patch: str):
        # Discard = measure and forget, or deallocate.
        # Placeholder: measure all data qubits in Z and do nothing with the results.
        qs = self.pm.qubits(patch)
        self.c.append("M", qs)
        self.c.append("TICK")

    def append_y_meas_stub(self, patch: str):
        # Placeholder for "logical Y measurement of a patch":
        # measure all data qubits in Y and parity them in software (not modeled).
        # We'll just measure one representative qubit in Y for wiring conditionals.
        q = self.pm.qubits(patch)[0]
        self.c.append("MY", [q])
        # Record index: stim uses rec[-1] to refer to last measurement.
        # We'll store the absolute index via counting is annoying; easiest is store "rec[-1]" usage.
        self.last_y_meas_rec[patch] = -1
        self.c.append("TICK")

    def append_cond_s_from_y(self, patch_target: str, meas_patch: str):
        # This models: if Y measurement == 1, apply S (or S_DAG) to target patch.
        # Stim supports classical control via "CX rec[...] q" style and also via blocks.
        #
        # We'll use a loop-free conditional:
        #   - Convert measurement record into a Z correction using feedback gadgets is tricky;
        #   - But for a starter, use "CORRELATED_ERROR" isn't conditional.
        #
        # Best simple way: use stim's classical control on Pauli gates:
        #   e.g., "CZ rec[-1] q" is allowed for some ops? Actually Stim supports
        #   measurement-controlled Pauli ops like "CX rec[-1] q", "CZ rec[-1] q".
        #   There isn't a direct controlled-S.
        #
        # So for a *starter*:
        # - Track the conditional S in your own classical side data structure (Pauli frame / Clifford frame),
        #   OR
        # - Approximate conditional S by updating a frame instead of applying it in-circuit.
        #
        # We'll do the recommended approach: frame tracking hook.
        pass

    def translate(self, schedule: List[Instr]) -> stim.Circuit:
        for ins in schedule:
            if ins.op == "IDLE":
                self.append_idle(ins.patch, ins.rounds)
            elif ins.op == "INJECT_T":
                self.append_inject_t_stub(ins.patch)
            elif ins.op == "MERGE":
                self.append_merge_stub(ins.patch, ins.patch2, ins.rounds)
            elif ins.op == "Y_MEAS":
                self.append_y_meas_stub(ins.patch)
            elif ins.op == "DISCARD":
                self.append_discard(ins.patch)
            else:
                raise ValueError(f"Unknown op {ins.op}")
        return self.c

# ----------------------------
# Example usage
# ----------------------------
pm = PatchMap()
pm.alloc_patch("A", n_data=9)  # toy "patch A"
pm.alloc_patch("B", n_data=9)  # toy "patch B" (neighbor)

sched = [
    Instr("IDLE", patch="A", rounds=1),
    Instr("INJECT_T", patch="B"),
    Instr("MERGE", patch="A", patch2="B", rounds=1),
    Instr("DISCARD", patch="B"),
    # COND_S decomposed:
    Instr("MERGE", patch="A", patch2="B", rounds=1),
    Instr("Y_MEAS", patch="B"),
    Instr("IDLE", patch="A", rounds=1),
    Instr("DISCARD", patch="B"),
]

translator = LatticeSurgeryToStim(pm)
circuit = translator.translate(sched)
print(circuit)
