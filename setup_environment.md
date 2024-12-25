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
pip install transformers==4.46.0
pip install accelerate==0.26.0
pip install protobuf==3.20.*
pip install --upgrade Pillow
```
## Other packages
```bash
cd ..
pip install -r requirements.txt
pip install faiss-cpu
pip install faiss-gpu-cuxx # depends on your CUDA version
```