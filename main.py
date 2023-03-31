import argparse
import time
from functools import partial
from typing import Callable, Dict, Iterable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

from nbody.utils import setup_nbody_data
from egnn_jax.variants import EGNN_vel

key = jax.random.PRNGKey(1337)


@partial(jax.jit, static_argnames=["model_fn"])
def mse(
    params: hk.Params,
    graph: jraph.GraphsTuple,
    props: Dict[str, jnp.ndarray],
    target: jnp.ndarray,
    model_fn: Callable,
) -> Tuple[float]:
    pred = model_fn(
        params,
        graph,
        **props,
    )
    assert target.shape == pred.shape
    return (jnp.power(pred - target, 2)).mean()


@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params: hk.Params,
    graph: jraph.GraphsTuple,
    props: Dict[str, jnp.ndarray],
    target: jnp.ndarray,
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, hk.Params, optax.OptState]:
    loss, grads = jax.value_and_grad(loss_fn)(params, graph, props, target)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state


def evaluate(
    loader: Iterable,
    params: hk.Params,
    loss_fn: Callable,
    graph_transform: Callable,
) -> Tuple[float, float]:
    eval_loss = 0.0
    eval_times = 0.0
    for data in loader:
        graph, props, target = graph_transform(data)
        eval_start = time.perf_counter_ns()
        loss = jax.lax.stop_gradient(loss_fn(params, graph, props, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6

    return eval_times / len(loader), eval_loss / len(loader)


def train(egnn, loader_train, loader_val, loader_test, graph_transform, args):
    init_graph, init_props, _ = graph_transform(next(iter(loader_train)))
    params = egnn.init(
        key,
        init_graph,
        **init_props
    )

    print(
        f"Starting {args.epochs} epochs on {args.dataset} "
        f"with {hk.data_structures.tree_size(params)} parameters.\n"
        "Jitting..."
    )

    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    loss_fn = partial(mse, model_fn=egnn.apply)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)
    avg_time = []
    best_val = 1e10

    for e in range(args.epochs):
        train_loss = 0.0
        train_start = time.perf_counter_ns()
        for data in loader_train:
            graph, props, target = graph_transform(data)
            loss, params, opt_state = update_fn(
                params=params,
                graph=graph,
                props=props,
                target=target,
                opt_state=opt_state,
            )
            train_loss += loss
        train_time = (time.perf_counter_ns() - train_start) / 1e6
        train_loss /= len(loader_train)
        print(
            f"[Epoch {e+1:>4}] train loss {train_loss:.6f}, epoch {train_time:.2f}ms",
            end="",
        )
        if e % args.val_freq == 0:
            eval_time, val_loss = eval_fn(loader_val, params)
            avg_time.append(eval_time)
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                _, test_loss_ckp = eval_fn(loader_test, params)
                tag = " (BEST)"
            print(f" - val loss {val_loss:.6f}{tag}, infer {eval_time:.2f}ms", end="")

        print()

    test_loss = 0
    _, test_loss = eval_fn(loader_test, params)
    # ignore compilation time
    avg_time = avg_time[2:]
    avg_time = sum(avg_time) / len(avg_time)
    print(
        "Training done.\n"
        f"Final test loss {test_loss:.6f} - checkpoint test loss {test_loss_ckp:.6f}.\n"
        f"Average (model) eval time {avg_time:.2f}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model options
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--tanh", action="store_true")

    # data options
    parser.add_argument(
        "--dataset", type=str, default="charged", choices=["charged", "gravity"]
    )
    parser.add_argument(
        "--dataset-name", type=str, default="small", choices=["small", "default"]
    )
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--target", type=str, default="pos", choices=["pos", "force"])

    # training options
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-12)
    parser.add_argument("--val-freq", type=int, default=10)

    args = parser.parse_args()

    loader_train, loader_val, loader_test, graph_transform = setup_nbody_data(args)

    egnn = lambda graph, pos, vel, edge_attribute: EGNN_vel(
        hidden_size=args.hidden_size,
        output_size=args.hidden_size,
        num_layers=args.num_layers,
        residual=True,
        normalize=args.normalize,
        tanh=args.tanh,
    )(graph, pos, vel, edge_attribute)

    egnn = hk.without_apply_rng(hk.transform(egnn))

    train(egnn, loader_train, loader_val, loader_test, graph_transform, args)
