from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from egnn_jax import EGNN, EGNNLayer
from egnn_jax.utils import LinearXav


class EGNNLayer_vel(EGNNLayer):
    def __init__(
        self,
        hidden_size: int,
        *args,
        blocks: int = 1,
        act_fn: Callable = jax.nn.relu,
        dt: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            *args,
            hidden_size=hidden_size,
            blocks=blocks,
            act_fn=act_fn,
            dt=dt,
            **kwargs,
        )
        # velocity integrator network
        net = [LinearXav(hidden_size) for _ in range(blocks)]
        net += [
            act_fn,
            LinearXav(1, with_bias=False, w_init=hk.initializers.UniformScaling(dt)),
        ]
        self._vel_correction_mlp = hk.Sequential(net)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        graph, pos = super().__call__(graph, pos, edge_attribute, node_attribute)
        shift = self._vel_correction_mlp(graph.nodes) * vel
        pos = pos + jnp.clip(shift, -100, 100)
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
        h = LinearXav(self._hidden_size, name="embedding")(graph.nodes)
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
