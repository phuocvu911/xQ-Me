QASM Peak-State Finder
======================

This small project loads a QASM description from a file, constructs a Qiskit
circuit, computes the output statevector, and prints the most probable
bitstring (with its probability).

Requirements
------------
- Python 3.8+ (this repo was tested with Python 3.14)
- qiskit, qiskit-aer, numpy

Install
-------
Use `python3` and `pip3` to install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install qiskit qiskit-aer numpy
```

Usage
-----
1. Place your QASM file next to `main.py` and name it `qasm.txt` (or update
   the path in `main.py`).
2. Run the script with:

```bash
python3 main.py
```

Reading QASM inside `main.py`
-----------------------------
If `main.py` doesn't already load the QASM file, add this snippet before
creating the circuit to read `qasm.txt` from the same directory:

```python
from pathlib import Path
qasm = Path(__file__).resolve().parent.joinpath('qasm.txt').read_text(encoding='utf-8')
```

What the script prints
-----------------------
The script prints the most likely output bitstring and its probability, e.g.

```
Answer: 010 (prob=0.4231)
```

Next steps
----------
- Edit `main.py` to point to a different QASM file if needed.
- Add example `qasm.txt` files for testing.
