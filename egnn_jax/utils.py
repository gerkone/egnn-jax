from typing import Iterable, Optional

import haiku as hk


class LinearXav(hk.Linear):
    """Linear layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        super().__init__(output_size, with_bias, w_init, b_init, name)


class MLPXav(hk.nets.MLP):
    """MLP layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        activation: Optional[hk.initializers.Initializer] = None,
        activate_final: bool = False,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        if not with_bias:
            b_init = None
        super().__init__(
            output_sizes,
            w_init,
            b_init,
            with_bias,
            activation,
            activate_final,
            name,
        )
