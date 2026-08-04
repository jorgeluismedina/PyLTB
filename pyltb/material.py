
import numpy as np

class Material:
    def __init__(self, E, nu, rho):
        self.E   = E
        self.nu  = nu
        self.rho = rho
        self.G   = E / (2 * (1 + nu))