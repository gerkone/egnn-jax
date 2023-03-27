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
            activate_final=False,
        )

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        graph, pos = super().__call__(graph, pos, edge_attribute, node_attribute)
        pos += self._vel_correction_mlp(graph.nodes) * vel
        return graph, pos


class EGNN_vel(EGNN):
    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # input node embedding
        h = hk.Linear(self._hidden_size, name="embedding")(graph.nodes)
        graph = graph._replace(nodes=h)
        # message passing
        for n in range(self._num_layers):
            graph, pos = EGNNLayer_vel(
                layer_num=n,
                hidden_size=self._hidden_size,
                output_size=self._hidden_size,
                act_fn=self._act_fn,
                residual=self._residual,
                attention=self._attention,
                normalize=self._normalize,
                tanh=self._tanh,
            )(graph, pos, vel, edge_attribute, node_attribute)
        return pos
