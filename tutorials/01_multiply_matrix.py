from jax import random

if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))

    print(x)
