
import numpy as np

class Material:
    def __init__(self, E, nu, dens):
        self.E = E
        self.nu = nu
        self.dens = dens
        self.G = E / (2 * (1 + nu))