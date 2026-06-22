
import numpy as np

class Beam():
    def __init__(self, mater, coords, conec, vrx_dofs, ltr_dofs):
        self.mater  = mater
        self.coords = coords 
        self.conec  = conec
        self.vrx_dofs = vrx_dofs # vertical deflection and axial displacement DOFs
        self.ltr_dofs = ltr_dofs # lateral deflection and torsion DOFs
    


