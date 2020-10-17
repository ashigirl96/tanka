from typing import Generator, Mapping

import jax
import jax.numpy as np
import tensorflow_datasets as tfds
from chex import Numeric


def softmax_cross_entropy(logits: Numeric, labels: Numeric):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -np.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(predicts, labels):
    return np.mean(softmax_cross_entropy(predicts, labels))


def sgd(param, update):
    return param - 0.01 * update


Batch = Mapping[str, np.ndarray]


def load_dataset(split: str, batch_size: int) -> Generator[Batch, None, None]:
    # ds = tfds.load("binarized_mnist", split=split, shuffle_files=True)
    ds = tfds.load("mnist", split=split, shuffle_files=True)
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


if __name__ == "__main__":
    # loss_fn_t = hk.transform(loss_fn)
    # loss_fn_t = hk.without_apply_rng(loss_fn_t)
    #
    # rng = jax.random.PRNGKey(42)
    #
    # for image, labels in input_dataset:
    #     grads = jax.grad(loss_fn_t.apply)(params, )
    train_ds = load_dataset(tfds.Split.TRAIN, 32)

    # for x in train_ds:
    #     print(x.)
    sample = next(train_ds)
    print(sample["image"].shape)
