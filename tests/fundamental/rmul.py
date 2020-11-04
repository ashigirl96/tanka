class Op:
    def __init__(self, name: str):
        self.name = name

    def __mul__(self, other: str):
        return self.name + other

    def __rmul__(self, other: str):
        return self.name + other

    def __add__(self, other):
        return self.name + other

    def __radd__(self, other):
        return self.name + other


x = Op("1")
print(x * "2")
print("2" * x)

print(x + "3")
print("3" + x)
