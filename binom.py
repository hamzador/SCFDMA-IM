import math as np

def binom(n, k):
    return np.factorial(n) // np.factorial(k) // np.factorial(n - k)