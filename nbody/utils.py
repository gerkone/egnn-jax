from typing import Callable, Dict, List, Tuple

import jax.numpy as jnp
import jraph
import numpy as np
from torch.utils.data import DataLoader

from .datasets import NBodyDataset


def NbodyGraphTransform(
    n_nodes: int,
    batch_size: int,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    # charged system is a connected graph
    full_edge_indices = jnp.array(
        [
            (i + n_nodes * b, j + n_nodes * b)
            for b in range(batch_size)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        ]
    ).T

    def _to_jraph(
        data: List,
    ) -> Tuple[jraph.GraphsTuple, Dict[str, jnp.ndarray], jnp.ndarray]:
        props = {}
        pos, vel, edge_attribute, _, targets = data

        cur_batch = int(pos.shape[0] / n_nodes)

        edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
        senders, receivers = edge_indices[0], edge_indices[1]

        # relative distances between particles
        pos_dist = jnp.sum((pos[senders] - pos[receivers]) ** 2, axis=-1)[:, None]
        props["edge_attribute"] = jnp.concatenate([edge_attribute, pos_dist], axis=-1)
        props["pos"] = pos
        props["vel"] = vel

        graph = jraph.GraphsTuple(
            # velocity magnitude as node features (scalar)
            nodes=jnp.sqrt(jnp.sum(vel**2, axis=-1))[:, None],
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_nodes] * cur_batch),
            n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
            globals=None,
        )

        return (
            graph,
            props,
            targets,
        )

    return _to_jraph


def numpy_collate(batch):
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.vstack(batch)


def setup_nbody_data(args) -> Tuple[DataLoader, DataLoader, DataLoader, Callable]:
    dataset_train = NBodyDataset(partition="train", max_samples=args.max_samples)
    dataset_val = NBodyDataset(partition="val")
    dataset_test = NBodyDataset(partition="test")

    graph_transform = NbodyGraphTransform(n_nodes=5, batch_size=args.batch_size)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )

    return loader_train, loader_val, loader_test, graph_transform
