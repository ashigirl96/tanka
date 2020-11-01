import weakref

import jax.numpy as np

a = np.array([1, 2, 3], dtype=np.float64)
b = weakref.ref(a)

print(b)
print(weakref.getweakrefcount(a))
print(b())

a = None
print(weakref.getweakrefcount(a))
print(b)
