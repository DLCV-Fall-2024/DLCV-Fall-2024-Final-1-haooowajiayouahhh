## Setup Environment
```bash
conda create -n myenv python=3.11
```
## LLaVA packages
```bash
cd LLaVA/
pip install -r llava_requirements.txt
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## Other packages
```bash
cd ..
pip install -r requirements.txt
pip install faiss-cpu
pip install faiss-gpu-cuxx # depends on your CUDA version
```