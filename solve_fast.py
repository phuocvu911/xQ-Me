#!/usr/bin/env python3
"""
Fast peaked-circuit solver via direct MPS sampling.
Skips deobfuscation (slow transpile) and computes marginals from shot counts.
"""
import sys, re, math, os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── QASM loader (same preprocessing as main.py) ───────────────────────────────
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


# ── Strategy 1: MPS sampling ──────────────────────────────────────────────────
def solve_by_sampling(qc, shots=4096, bond_dim=32):
    """Run the circuit with MPS simulator, return most frequent bitstring."""
    n = qc.num_qubits
    print(f"[sampling] {n} qubits, {len(qc.data)} gates, bond_dim={bond_dim}, shots={shots}", flush=True)

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        matrix_product_state_truncation_threshold=1e-10,
    )
    c = qc.copy()
    c.measure_all()
    c = transpile(c, sim, optimization_level=0)  # no optimisation = fast

    counts = sim.run(c, shots=shots).result().get_counts()
    top = sorted(counts.items(), key=lambda x: -x[1])

    print(f"[sampling] top-5 outcomes:")
    for bs, cnt in top[:5]:
        print(f"  {bs.replace(' ','')}  count={cnt}  prob={cnt/shots:.4f}")

    peak = top[0][0].replace(" ", "")
    peak_prob = top[0][1] / shots
    return peak, peak_prob


# ── Strategy 2: marginals via Z expectations (no deobf) ──────────────────────
def solve_by_marginals(qc, bond_dim=64):
    """Compute <Z_i> for every qubit via MPS expectation values."""
    from qiskit.quantum_info import Pauli
    n = qc.num_qubits
    print(f"[marginals] {n} qubits, bond_dim={bond_dim}", flush=True)

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

    bits = (z < 0).astype(int)
    conf = np.abs(z)
    # rightmost char = qubit 0 (challenge convention)
    bitstr = "".join(str(int(bits[i])) for i in reversed(range(n)))
    print(f"[marginals] min |<Z>| = {conf.min():.4f}  (1.0 = certain)")
    return bitstr, conf


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "circuit.qasm"

    qc = load_circuit(path)
    print(f"Loaded: {qc.num_qubits} qubits, {len(qc.data)} gates")

    # Try fast sampling first
    peak, prob = solve_by_sampling(qc, shots=4096, bond_dim=32)
    print(f"\nPEAK BITSTRING (sampling): {peak}")
    print(f"estimated probability: {prob:.4f}")

    if prob < 0.05:
        # Low peak probability — escalate to marginal method with higher bond
        print("\n[escalating to marginals with bond_dim=128]")
        peak2, conf = solve_by_marginals(qc, bond_dim=128)
        print(f"\nPEAK BITSTRING (marginals): {peak2}")
        print(f"min confidence: {conf.min():.4f}")
