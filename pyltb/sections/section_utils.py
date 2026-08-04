

import numpy as np
from pyltb.sections.section_bs import ISection_BS
from pyltb.sections.section_ms import ISection_MS


def interpolate_section(section1, section2, xi):
    """
    Interpola linealmente las dimensiones geometricas entre dos secciones.

    Retorna una nueva ISection_MS con propiedades interpoladas en xi ∈ [0, 1].
    xi = 0 → seccion identica a section1
    xi = 1 → seccion identica a section2
    """
    h   = section1.h   + (section2.h   - section1.h)   * xi
    bf1 = section1.bf1 + (section2.bf1 - section1.bf1) * xi
    bf2 = section1.bf2 + (section2.bf2 - section1.bf2) * xi
    tw  = section1.tw  + (section2.tw  - section1.tw)  * xi
    tf1 = section1.tf1 + (section2.tf1 - section1.tf1) * xi
    tf2 = section1.tf2 + (section2.tf2 - section1.tf2) * xi
    r1  = section1.r1  + (section2.r1  - section1.r1)  * xi
    r2  = section1.r2  + (section2.r2  - section1.r2)  * xi

    return ISection_MS(h=h, bf1=bf1, bf2=bf2, tw=tw,
                       tf1=tf1, tf2=tf2, r1=r1, r2=r2)


def interpolate_multiple_sections(section1, section2, points):
    return [interpolate_section(section1, section2, xi) for xi in points]



def build_mesh(L, breakpoints, sections, nelems, etype=1, mat_id=0):
    """
    Nodos, secciones y conectividad de una viga con seccion variable por tramos,
    interpolando linealmente entre secciones de control `sections_cp` ubicadas
    en posiciones normalizadas `breakpoints` (0 a 1).

    Parameters
    ----------
    L : float                       Longitud total.
    breakpoints : array_like        Posiciones normalizadas [0,1], crecientes, incluye 0.0 y 1.0.
    sections    : list[ISection_MS] Secciones en cada breakpoint (mismo largo que breakpoints).
    nelems : int                    Numero de elementos.
    etype, mat_id : int             Constantes para elements_data (mismo tipo/material en toda la malla).

    Returns
    -------
    coords        : ndarray (nelems+1, 1)
    sections      : list[ISection_MS], largo nelems+1 (una por nodo)
    elements_data : ndarray (nelems, 4)  — [etype, mat_id, nodei, nodej]
    """
    breakpoints = np.asarray(breakpoints, dtype=float)
    xi = np.linspace(0.0, 1.0, nelems + 1)

    s = np.interp(xi, breakpoints, np.arange(len(breakpoints)))  # coordenada continua de tramo
    k = np.minimum(s.astype(int), len(breakpoints) - 2)          # parte entera → indice de tramo
    t = s - k                                                    # parte fraccionaria → t

    sections = [interpolate_section(sections[ki], sections[ki + 1], ti)
                for ki, ti in zip(k, t)]
    
    #coords = (xi * L).reshape(-1, 1)
    coords = (xi * L)

    elements_data = np.column_stack([
        np.full(nelems, etype), 
        np.full(nelems, mat_id),
        np.arange(nelems), 
        np.arange(1, nelems + 1)
    ])

    return coords, sections, elements_data