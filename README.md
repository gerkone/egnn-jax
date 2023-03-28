# E(n) Equivariant GNN in jax
Reimplementation of [EGNN](https://arxiv.org/abs/2102.09844) in jax. Original work by Victor Garcia Satorras, Emiel Hogeboom, Max Welling.

## Installation
```
python -m pip install egnn-jax
```

Or clone this repository and build locally
```
python -m pip install -e .
```


### GPU support
Upgrade `jax` to the gpu version
```
pip install --upgrade "jax[cuda]==0.4.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Validation
N-body (charged) is included for validation from the original paper. Times are  __model only__ on batches of 100 graphs, in (global) single precision.
|                  |  MSE  | Inference [ms]* |
|------------------|-------|-----------------|
| torch (original) | .0071 |      0.94       |
| jax (ours)       |       |                 |

\* remeasured (Quadro RTX 4000)

### Validation install

The N-Body experiments are only included in the github repo, so it needs to be cloned first.
```
git clone https://github.com/gerkone/egnn-jax
```

They are adapted from the original implementation, so additionally `torch` and `torch_geometric` are needed (cpu versions are enough).
```
pip3 install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r nbody/requirements.txt
```

### Valdation usage
The charged N-body dataset has to be locally generated in the directory [/nbody/data](/nbody/data).
```
python3 -u generate_dataset.py --num-train=3000
```
Then, the model can be trained and evaluated (from the repo root) with
```
python main.py --epochs=200 --batch-size=100 --lr=5e-3 --weight-decay=1e-12
```


## Acknowledgements
This implementation is heavily inspired from the [original pytorch code](https://github.com/vgsatorras/egnn).
