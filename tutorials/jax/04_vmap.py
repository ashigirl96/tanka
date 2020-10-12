from jax import jit, random, vmap

from tutorials.utility.timer import print_timer

if __name__ == "__main__":
    batch_size = 5
    key = random.PRNGKey(43)
    mat = random.normal(key, (150, 100))
    batched_x = random.normal(key, (batch_size, 100))

    def apply_matrix(x):
        return mat.dot(x)

    @jit
    def vmap_batched_apply_matrix(v_batched):
        return vmap(apply_matrix)(v_batched)

    print_timer(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready())
