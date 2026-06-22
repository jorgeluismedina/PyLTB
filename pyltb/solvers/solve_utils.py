
import numpy as np
import scipy as sp


def check_symmetric(a, rtol=1e-6, atol=1e-5):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)



# Para elemento isoparametrico shell de 8 nodos.
def detailed_check_SPD(Kg): # matriz semidefinida positiva
    eigs = sp.linalg.eigh(Kg)
    
    print(f"Autovalor mínimo: {np.min(eigs):.2e}")
    print(f"Autovalor máximo: {np.max(eigs):.2e}")
    
    # Contar autovalores según su magnitud
    zero_eigs = np.sum(np.abs(eigs) < 1e-10)  # Prácticamente cero
    positive_eigs = np.sum(eigs > 1e-10)      # Claramente positivos
    negative_eigs = np.sum(eigs < -1e-10)     # Claramente negativos
    
    print(f"Autovalores ~0 (modos cuerpo rígido): {zero_eigs}")
    print(f"Autovalores >0 (modos elásticos): {positive_eigs}") 
    print(f"Autovalores <0 (problemas!): {negative_eigs}")
    
    # Para un elemento shell de 8 nodos libre, esperamos:
    # - 6 autovalores cero (modos cuerpo rígido)
    # - 42 autovalores positivos (modos elásticos)
    # - 0 autovalores negativos
    
    expected_rigid_body_modes = 6
    if zero_eigs == expected_rigid_body_modes and negative_eigs == 0:
        print("✓ ¡PERFECTO! El elemento tiene el comportamiento esperado")
    else:
        print("✗ Comportamiento inesperado - revisar implementación")
