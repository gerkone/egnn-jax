from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from egnn_jax import EGNN, EGNNLayer


class EGNNLayer_vel(EGNNLayer):
    def __init__(
        self,
        hidden_size: int,
        *args,
        blocks: int = 1,
        act_fn: Callable = jax.nn.relu,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size, blocks=blocks, act_fn=act_fn, *args, **kwargs
        )

        self._vel_correction_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [1],
            activation=act_fn,
            activate_final=True,
        )

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        node_attribute: Optional[jnp.ndarray] = None,
        edge_attribute: Optional[jnp.ndarray] = None,
    ):
        super().__call__(graph, pos, node_attribute, edge_attribute)
        pos += self._vel_correction_mlp(graph.nodes) * vel
        return graph, pos


class EGNN_vel(EGNN):
    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        node_attribute: Optional[jnp.ndarray] = None,
        edge_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _, x = super().__call__(graph, pos, node_attribute, edge_attribute)
        return x
