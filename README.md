# E(n) Equivariant GNN in jax
Reimplementation of [EGNN](https://arxiv.org/abs/2102.09844) in jax. Original work by Victor Garcia Satorras, Emiel Hogeboom and Max Welling.

## Installation
```
python -m pip install egnn-jax
```

Or clone this repository and build locally
```
git clone https://github.com/gerkone/egnn-jax
cd painn-jax
python -m pip install -e .
```
### GPU support
Upgrade `jax` to the gpu version
```
pip install --upgrade "jax[cuda]==0.4.10" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Validation
N-body (charged) is included for validation from the original paper. Times are  __model only__ on batches of 100 graphs, in (global) single precision.
|                  |  MSE  | Inference [ms]* |
|------------------|-------|-----------------|
| [torch (original)](https://github.com/vgsatorras/egnn) | .0071 |      8.27       |
| jax (ours)       | .0084 |      0.94       |

\* remeasured (Quadro RTX 4000)

### Validation install

The N-Body experiments are only included in the github repo, so it needs to be cloned first.
```
git clone https://github.com/gerkone/egnn-jax
```

They are adapted from the original implementation, so additionally `torch` and `torch_geometric` are needed (cpu versions are enough).
```
python -m pip install -r nbody/requirements.txt
```

### Valdation usage
The charged N-body dataset has to be locally generated in the directory [/nbody/data](/nbody/data).
```
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix small
```
Then, the model can be trained and evaluated with
```
python validate.py --epochs=1000 --batch-size=100 --lr=1e-4 --weight-decay=1e-12
```

## Acknowledgements
This implementation heavily borrows from the [original pytorch code](https://github.com/vgsatorras/egnn).
