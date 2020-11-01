import sys


class obj:
    pass


def f(x):
    print(sys.getrefcount(x) - 1)


a = obj()
assert sys.getrefcount(a) - 1 == 1
f(a)
assert sys.getrefcount(a) - 1 == 1
a = None
print(sys.getrefcount(a) - 1)

a = obj()
b = obj()
c = obj()

a.b = b
b.c = c

print(sys.getrefcount(a) - 1)
print(sys.getrefcount(b) - 1)
print(sys.getrefcount(c) - 1)
a = b = c = None
print(sys.getrefcount(a) - 1)
print(sys.getrefcount(b) - 1)
print(sys.getrefcount(c) - 1)

a = obj()
b = obj()
c = obj()
a.b = b
b.c = c
c.a = a

print(sys.getrefcount(a) - 1)
print(sys.getrefcount(b) - 1)
print(sys.getrefcount(c) - 1)
a = b = c = None
print(sys.getrefcount(a) - 1)
print(sys.getrefcount(b) - 1)
print(sys.getrefcount(c) - 1)
