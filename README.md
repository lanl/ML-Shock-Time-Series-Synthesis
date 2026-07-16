# ML Shock Time Series Synthesis

Repository name: `gen_srs_public`

LANL copyright assertion number: **O5074**

This repository provides three core capabilities:

1. **GPU-batched synthetic shock time-series generation at scale** (thousands to hundreds of thousands of shocks): choose a sample rate and basis function, then generate the corresponding shock waveforms.
2. **GPU-batched Shock Response Spectrum (SRS) computation** to efficiently produce matched time-series/SRS pairs.
3. **Four benchmark shock time-series datasets** spanning multiple domains, pre-cleaned and standardized (**32,768 Hz** sample rate; ~**9K samples per time series**).



## Getting started

```
git clone https://git.lanl.gov/srs_synth/gen_srs_public.git
cd gen_srs_public
```

## Installation

Create and activate a Conda environment with Python `3.12.12` (tested), then install the package dependencies. Newer Python versions may also work, but they have not been officially tested yet:

```
conda create -n <env_name> python=3.12.12
conda activate <env_name>
conda install pip
pip install -r <requirements.txt or requirements.cuda.txt>
pip install -e .
```

Use `requirements.txt` for CPU installs, including macOS (`OSX`) systems. Use `requirements.cuda.txt` on systems with NVIDIA GPUs and CUDA support.

## Benchmark datasets
Fully open-source: LA-UR-26-23746

The `.npz` files under `Datasets/` should be treated as release assets, not bundled package data.

- In a repo checkout, dataset downloads default to the local `Datasets/` folder.
- In an installed environment, dataset downloads default to a user-writable cache directory.
- You can override the download location with `GEN_SRS_DATA_DIR`.
- You should set `GEN_SRS_RELEASE_REPO`, and optionally `GEN_SRS_RELEASE_OWNER` and `GEN_SRS_RELEASE_TAG`, so `gen_srs.datasets` can fetch the release assets.

If the package is installed, you can download all benchmark datasets with:

```bash
gen-srs-datasets
```

## License

The OSS copyright notice is provided in [License.txt](License.txt).

This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

- Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

(End of Notice)
