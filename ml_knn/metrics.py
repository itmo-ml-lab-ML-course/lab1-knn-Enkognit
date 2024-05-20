import math


def gauss_kernel(x):
    return math.exp(- x * x / 2) / math.sqrt(2 * math.pi)

def uniform_kernel(x):
    return 1/2 if abs(x) < 1 else 0

def epanechnikov_kernel(x):
    return 3/4 * (1 - x * x) if abs(x) < 1 else 0

def triangular_kernel(x):
    return (1 - abs(x)) if abs(x) < 1 else 0

def kernel_factory(a, b):
    def kernel(x):
        return math.pow(1 - math.pow(abs(x), a), b) if abs(x) < 1 else 0
    return kernel
