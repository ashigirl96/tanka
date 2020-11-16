import contextlib


@contextlib.contextmanager
def fn(x):
    print("Start")
    try:
        print(x)
        yield
    finally:
        print("End")


if __name__ == "__main__":
    with fn("Hello"):
        print("Process...")
