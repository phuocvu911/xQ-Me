#!/usr/bin/env python3
"""
solve_v2.py — peaked circuit solver with fast deobfuscation.

Key insight: the circuit is obfuscated with U·U† pairs that produce high
entanglement when simulated raw, but collapse to a near-product state after
gate cancellation. optimization_level=1 (fast) merges consecutive single-qubit
rotations and cancels CX pairs in O(n·d) time — far cheaper than level 3.
"""
import sys, re, math, os, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

# ── QASM loader ───────────────────────────────────────────────────────────────
def _find_qelib1():
    import qiskit
    for root, dirs, files in os.walk(os.path.dirname(qiskit.__file__)):
        if "qelib1.inc" in files:
            return os.path.join(root, "qelib1.inc")
    return None

_QELIB1_PATH = _find_qelib1()
_QELIB1_CONTENT = open(_QELIB1_PATH).read() if _QELIB1_PATH else ""
_EXTRA_GATES = """
gate rzx(theta) a,b { h b; cx a,b; rz(theta) b; cx a,b; h b; }
gate ecr a,b { rzx(pi/4) a,b; x a; rzx(-pi/4) a,b; }
gate iswap a,b { swap a,b; s a; s b; h a; cx a,b; h b; }
gate dcx a,b { cx a,b; cx b,a; }
gate zz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }
"""

def _fix_qasm(txt):
    if _QELIB1_CONTENT:
        txt = re.sub(r'include\s+"qelib1\.inc"\s*;', _QELIB1_CONTENT, txt, count=1)
    if "gate rzx" not in txt:
        txt = re.sub(r'(qreg\s)', _EXTRA_GATES + r'\1', txt, count=1)
    safe = {"pi": math.pi, "__builtins__": {}}
    def _eval_pi(m):
        try:
            return "({:.17g})".format(eval(m.group(1), safe))
        except Exception:
            return m.group(0)
    txt = re.sub(r"\(([^()]*\bpi\b[^()]*)\)", _eval_pi, txt)
    return txt

def load_circuit(path):
    txt = _fix_qasm(open(path).read())
    import qiskit.qasm2 as q2
    qc = q2.loads(txt)
    return qc.remove_final_measurements(inplace=False)


# ── fast deobfuscation (level 1 only — ~10x faster than level 3) ──────────────
def fast_deobf(qc):
    t0 = time.time()
    # Level 1: merges adjacent single-qubit rotations, cancels back-to-back CX
    result = transpile(
        qc,
        optimization_level=1,
        basis_gates=["rx", "ry", "rz", "h", "cx"],
    )
    print(f"[deobf L1]  {len(qc.data)} → {len(result.data)} gates  ({time.time()-t0:.1f}s)", flush=True)
    return result


# ── Z-marginal solver (no shots, single deterministic pass) ───────────────────
def marginals(qc, bond_dim=32):
    n = qc.num_qubits
    t0 = time.time()
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        matrix_product_state_truncation_threshold=1e-12,
    )
    c = qc.copy()
    for i in range(n):
        c.save_expectation_value(Pauli("Z"), [i], label=f"z{i}")
    c = transpile(c, sim, optimization_level=0)
    d = sim.run(c).result().data(0)
    z = np.array([float(np.real(d[f"z{i}"])) for i in range(n)])
    print(f"[marginals] bond={bond_dim}  min|<Z>|={np.abs(z).min():.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return z


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "circuit.qasm"
    t_start = time.time()

    qc = load_circuit(path)
    n = qc.num_qubits
    print(f"Loaded: {n} qubits, {len(qc.data)} gates")

    # Step 1: fast deobfuscation
    qc_simple = fast_deobf(qc)

    # Step 2: marginals at low bond dimension (enough for near-product state)
    z = marginals(qc_simple, bond_dim=32)
    bits = (z < 0).astype(int)
    conf = np.abs(z)
    bitstr = "".join(str(int(bits[i])) for i in reversed(range(n)))

    weak = np.where(conf < 0.5)[0]
    if len(weak):
        print(f"[warn] {len(weak)} low-confidence bits (|<Z>|<0.5): {list(weak)}")
        # Escalate only for the whole circuit if too many weak bits
        if len(weak) > n // 4:
            print("[escalate] re-running with bond_dim=128 ...")
            z = marginals(qc_simple, bond_dim=128)
            bits = (z < 0).astype(int)
            conf = np.abs(z)
            bitstr = "".join(str(int(bits[i])) for i in reversed(range(n)))

    print(f"\n{'='*50}")
    print(f"PEAK BITSTRING: {bitstr}")
    print(f"min |<Z>| confidence: {conf.min():.4f}")
    print(f"total time: {time.time()-t_start:.1f}s")
