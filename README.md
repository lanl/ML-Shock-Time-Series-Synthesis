Library
/
README.md


# ML Shock Time Series Synthesis

**Python distribution:** `gen_srs_public`  
**LANL copyright assertion number:** **O5074**

This repository provides three core capabilities:

1. **GPU-batched synthetic shock time-series generation at scale.** Generate thousands to hundreds of thousands of shock waveforms using a selected sample rate and basis function.
2. **GPU-batched Shock Response Spectrum (SRS) computation.** Efficiently produce matched time-series and SRS pairs.
3. **Four benchmark shock time-series datasets.** The datasets span multiple domains and are pre-cleaned and standardized to a **32,768 Hz** sample rate and **9,000 samples per time series**.

## Getting started

```bash
git clone https://github.com/lanl/ML-Shock-Time-Series-Synthesis.git
cd ML-Shock-Time-Series-Synthesis
```

## Installation

Python **3.12.12** is the officially tested version. The package metadata supports Python **3.11 or newer**. Newer Python versions may work but have not necessarily been tested.

### Linux and Windows with an NVIDIA GPU using CUDA 13.0

The CUDA requirements file installs the tested PyTorch CUDA 13.0 stack. General Python dependencies are installed separately from PyPI.

```bash
conda create -n <env_name> python=3.12.12
conda activate <env_name>
conda install pip

python -m pip install -r requirements-cu130.txt
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Verify that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### macOS

Install the PyTorch packages from PyPI, followed by the general dependencies and the local package. PyTorch will use Apple Metal Performance Shaders when supported by the system and selected by the application.

```bash
conda create -n <env_name> python=3.12.12
conda activate <env_name>
conda install pip

python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### Development and testing support

Install the package with the development extras to add `pytest` and `ipykernel`:

```bash
python -m pip install -e ".[dev]"
```

Testing support only:

```bash
python -m pip install -e ".[test]"
```

Notebook kernel support only:

```bash
python -m pip install -e ".[notebook]"
```

To register the active Conda environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"
```

## Benchmark datasets

The **SRS Benchmark Datasets v1.0.0**, approved for public release under **LA-UR-26-23746**, contain four compact, standardized collections for evaluating machine-learning methods on shock and transient time-series data: one public earthquake-derived dataset, one synthetic dataset, and two anonymized experimental datasets. Each contains multiple fixed-length signals sampled at **32,768 Hz**, with time in seconds and acceleration in G, and is suitable for reconstruction, spectral or wavelet analysis, regression, generative modeling, and synthetic-versus-real generalization studies.

Each benchmark is available as a wide-format CSV archive or as an NPZ archive optimized for NumPy, PyTorch, and `gen_srs_public`; both distributions include JSON metadata describing dimensions, units, sample rate, time range, data types, and source files. The NPZ datasets can be downloaded automatically through [`gen_srs_public/datasets.py`](gen_srs_public/datasets.py): calling `load_dataset("A")` downloads and extracts one requested release asset when it is not already cached, while calling `main()`—including by running `datasets.py` directly—or using `gen-srs-datasets` downloads all four datasets by default. Repository checkouts use `Datasets/` by default, installed packages use a user-writable cache, and `GEN_SRS_DATA_DIR` can override the destination.

## License

The open-source copyright notice is provided in [`License.txt`](License.txt).

This program is open source under the BSD 3-Clause License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS," AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT, INCLUDING NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
