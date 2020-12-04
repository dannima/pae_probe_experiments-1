1. Set `PYTHONPATH` as instructed in [https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/tree/master#setting-pythonpath](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/tree/master#setting-pythonpath).
 
2. Download `states-500000.ckpt` from [S3PRL Google Drive](https://drive.google.com/drive/folders/1d7nFh2I0J8EGdXJJ2_7zIeTcYzPtiX1Y), and store it in `checkpoints/`.

3. (optional) Install [apex](https://github.com/NVIDIA/apex#quick-start) for better speed when using PyTorch.

4. Run `./run.sh` to finish the probing experiments using Mockingjay features.