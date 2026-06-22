
"""
Funciones de forma de Hermite cúbicas para elementos de viga.
Dominio: ξ ∈ [0, 1]
"""
import numpy as np

def N_hermite(xi):
    """
    Funciones de forma de Hermite cúbicas.
    
    Retorna: [N1, N2, N3, N4]
    - N1, N3: funciones de traslación
    - N2, N4: funciones de rotación
    """
    xi2 = xi * xi
    xi3 = xi2 * xi
    
    N1 = 1 - 3*xi2 + 2*xi3
    N2 = xi - 2*xi2 + xi3
    N3 = 3*xi2 - 2*xi3
    N4 = -xi2 + xi3
    
    return np.array([N1, N2, N3, N4])


def dN_hermite(xi):
    """
    Primera derivada de funciones de Hermite respecto a ξ.
    
    Retorna: dN/dξ
    """
    xi2 = xi * xi
    
    dN1 = -6*xi + 6*xi2
    dN2 = 1 - 4*xi + 3*xi2
    dN3 = 6*xi - 6*xi2
    dN4 = -2*xi + 3*xi2
    
    return np.array([dN1, dN2, dN3, dN4])


def ddN_hermite(xi):
    """
    Segunda derivada de funciones de Hermite respecto a ξ.
    
    Retorna: d²N/dξ²
    """
    ddN1 = -6 + 12*xi
    ddN2 = -4 + 6*xi
    ddN3 = 6 - 12*xi
    ddN4 = -2 + 6*xi
    
    return np.array([ddN1, ddN2, ddN3, ddN4])



    
      



