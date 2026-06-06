from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np, time
from pathlib import Path
qasm = Path(__file__).resolve().parent.joinpath('qasm.txt').read_text(encoding='utf-8')

def solve(qasm_str, shots=50000, bond_dim=64):
    n = QuantumCircuit.from_qasm_str(qasm_str).num_qubits
    print(f"Solving {n}-qubit circuit...")

    if n <= 20:
        sv = Statevector.from_instruction(QuantumCircuit.from_qasm_str(qasm_str))
        probs = sv.probabilities()
        idx = int(np.argmax(probs))
        print(f"Answer: {format(idx, f'0{n}b')}  (exact, p={probs[idx]:.3f})")
        return format(idx, f'0{n}b')

    sim = AerSimulator(method='matrix_product_state',
                       matrix_product_state_max_bond_dimension=bond_dim,
                       matrix_product_state_truncation_threshold=1e-6)
    bits, weak = [], []
    t0 = time.time()
    for i in range(n):
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        cr = ClassicalRegister(1); qc.add_register(cr); qc.measure(i, cr[0])
        counts = sim.run(qc, shots=shots).result().get_counts()
        ones = sum(v for k, v in counts.items() if k.strip()[-1] == '1')
        p1 = ones / shots
        bits.append('1' if p1 > 0.5 else '0')
        margin = abs(p1 - 0.5) * 2          # 1.0 = certain, 0.0 = coin flip
        if margin < 0.3:
            weak.append((i, round(p1, 3)))

    # leftmost = q[n-1], rightmost = q[0]
    ans = ''.join(bits[i] for i in range(n - 1, -1, -1))
    print(f"Answer: {ans}   ({time.time()-t0:.0f}s)")
    if weak:
        print(f"Low-margin qubits (re-run with more shots if a submission fails): {weak}")
    return ans

solve(qasm)