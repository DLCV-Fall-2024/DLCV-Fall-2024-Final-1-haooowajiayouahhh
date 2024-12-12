## Setup Environment
```bash
1. conda create -n myenv python=3.11
```
## LLaVA packages
```bash
cd LLaVA/
2. pip install -r llava_requirements.txt
3. pip install -e .
4. pip install -e ".[train]"
5. pip install flash-attn --no-build-isolation
```
## Other packages
```bash
cd ..
6. pip install -r requirements.txt
7. pip install faiss-cpu
8. pip install faiss-gpu-cuxx # depends on your CUDA version
```