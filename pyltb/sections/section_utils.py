

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

