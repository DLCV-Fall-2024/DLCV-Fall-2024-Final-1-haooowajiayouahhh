# Setup Environment
1. conda create -n myenv python 3.11
### The following commands must follow this order!!!
2. pip install -r requirements.txt
3. pip install -e .
4. pip install -e ".[train]
5. pip install flash-attn --no-build-isolation
