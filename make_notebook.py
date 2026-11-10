"""Regenerate colab.ipynb from the files in src/ and the Makefile.

The notebook writes the sources with %%writefile, builds with nvcc and runs the benchmark,
so it works in Colab without cloning anything. Run this after editing any source file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["Makefile", "src/kernels.cuh", "src/main.cu"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md("# cuda-matmul: SGEMM from naive to vectorized\n\n"
       "Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.\n"),
    code("!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv\n!nvcc --version | tail -2"),
    code("!mkdir -p cuda-matmul/src"),
]
for rel in FILES:
    with open(os.path.join(HERE, rel)) as f:
        cells.append(code("%%writefile cuda-matmul/" + rel + "\n" + f.read()))
cells += [
    code("!cd cuda-matmul && make ARCH=native"),
    code("!cd cuda-matmul && ./sgemm 1024 2048 4096"),
    md("Copy the table above into `RESULTS.md` (GPU name, sizes, GFLOP/s, % of cuBLAS).\n\n"
       "Optional profiling: `!cd cuda-matmul && ncu --set basic -k regex:sgemm_2d ./sgemm 2048`"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(HERE, "colab.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote colab.ipynb")
