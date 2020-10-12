from timeit import timeit


def print_timer(func):
    print("{:.3f} ms".format(timeit(func, number=100)))
