from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import jraph
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_graph

from .datasets import ChargedDataset, GravityDataset


def NbodyGraphTransform(
    dataset_name: str,
    n_nodes: int,
    batch_size: int,
    neighbours: Optional[int] = 6,
    relative_target: bool = False,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    if dataset_name == "charged":
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

        if dataset_name == "charged":
            edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
            senders, receivers = edge_indices[0], edge_indices[1]
        if dataset_name == "gravity":
            batch = torch.arange(0, cur_batch)
            batch = batch.repeat_interleave(n_nodes).long()
            edge_indices = knn_graph(torch.from_numpy(np.array(pos)), neighbours, batch)
            # switched by default
            senders, receivers = jnp.array(edge_indices[0]), jnp.array(edge_indices[1])

        # relative distances among posations
        pos_dist = jnp.sum((pos[senders] - pos[receivers]) ** 2, axis=-1)[
            :, jnp.newaxis
        ]
        props["edge_attribute"] = jnp.concatenate([edge_attribute, pos_dist], axis=-1)
        props["pos"] = pos
        props["vel"] = vel

        graph = jraph.GraphsTuple(
            # velocity magnitude as node features (scalar)
            nodes=jnp.sqrt(jnp.sum(vel**2, axis=-1))[:, jnp.newaxis],
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_nodes] * cur_batch),
            n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
            globals=None,
        )
        # relative shift as target
        if relative_target:
            targets = targets - pos

        return (
            graph,
            props,
            targets,
        )

    return _to_jraph


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.vstack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


def setup_nbody_data(args) -> Tuple[DataLoader, DataLoader, DataLoader, Callable]:
    if args.dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
        )
        dataset_val = ChargedDataset(
            partition="val",
            dataset_name=args.dataset_name,
        )
        dataset_test = ChargedDataset(
            partition="test",
            dataset_name=args.dataset_name,
        )

    if args.dataset == "gravity":
        dataset_train = GravityDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            neighbours=args.neighbours,
            target=args.target,
        )
        dataset_val = GravityDataset(
            partition="val",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
        )
        dataset_test = GravityDataset(
            partition="test",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
        )

    graph_transform = NbodyGraphTransform(
        n_nodes=dataset_train.num_nodes,
        batch_size=args.batch_size,
        neighbours=args.neighbours,
        dataset_name=args.dataset,
    )

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
