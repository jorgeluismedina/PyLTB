
"""
Test sistemático para diagnosticar discrepancias en pandeo lateral‑torsional.
Compara PyLTB con LTBeamN variando alineamiento, tipo de carga y altura de aplicación.
Caso base: ménsula acartelada de 4 m, secciones bisimétricas, sin carga distribuida.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StabilityModel
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.sections.section_utils import interpolate_multiple_sections
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver

# ----------------------------------------------------------------------
material1 = Material(E=2.10e11, nu=0.3, dens=1.0)
materials = [material1]

# Secciones bisimétricas
section1 = ISection_MS(h=0.61, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)
section2 = ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)

# Secciones monosimétricas (ala superior más ancha)
section1_mono = ISection_MS(h=0.61, bf1=0.24, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)
section2_mono = ISection_MS(h=0.305, bf1=0.24, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)

L = 4.0
nelems = int(16 * L / 2)
nnods = nelems + 1
coordinates = np.linspace(0, L, nnods)
norm_coords = coordinates / L

node_sections_bi = interpolate_multiple_sections(section1, section2, norm_coords)
node_sections_mono = interpolate_multiple_sections(section1_mono, section2_mono, norm_coords)

elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])

verax_restraints = np.array([[0, 1, 1, 1]])
lator_restraints = np.array([[0, 1, 1, 1, 1]])

# ----------------------------------------------------------------------
# Casos bisimétricos (0-13)
cases_bi = [
    (0, -10.0e3, 0.0,     0, 0,  "Axial compresión en centroide"),
    (0, -10.0e3, 0.0,     3, 0,  "Axial compresión en ala superior"),
    (0, -10.0e3, 0.0,     2, 0,  "Axial compresión en ala inferior"),
    (0,   0.0,  -5.0e3,   0, 0,  "Transversal en centroide"),
    (0,   0.0,  -5.0e3,   0, 3,  "Transversal en ala superior"),
    (0, -10.0e3, -5.0e3,  3, 3,  "Combinada (Fx,Fz) en ala superior"),
    (3, -10.0e3, 0.0,     3, 0,  "Axial en ala sup"),
    (3, -10.0e3, 0.0,     0, 0,  "Axial en centroide"),
    (3,   0.0,  -5.0e3,   0, 3,  "Transversal en ala sup"),
    (3,   0.0,  -5.0e3,   0, 0,  "Transversal en centroide"),
    (3, -10.0e3, -5.0e3,  3, 3,  "Combinada en ala sup"),
    (2, -10.0e3, 0.0,     2, 0,  "Axial en ala inf"),
    (2,   0.0,  -5.0e3,   0, 2,  "Transversal en ala inf"),
    (2, -10.0e3, -5.0e3,  2, 2,  "Combinada (Fx,Fz) en ala inf"),
]

# Casos monosimétricos (14-18)
cases_mono = [
    (0, -10.0e3,  0.0,     0, 0,  "Axial compresión en centroide"),
    (0, -10.0e3,  0.0,     3, 0,  "Axial compresión en ala superior"),
    (0,   0.0,   -5.0e3,   0, 0,  "Transversal en centroide"),
    (3, -10.0e3,  0.0,     3, 0,  "Axial en ala sup"),
    (3, -10.0e3, -5.0e3,   3, 3,  "Combinada en ala sup"),
]

mu_cr_ltbeamn_bi = [
    31.55, 23.26, 23.26, 28.60, 8.859, 7.384,
    23.88, 31.34, 8.847, 28.56, 7.458,
    23.88, 43.78, 23.28
]
moments_ltbeamn_bi = [[0.0, 0.0], [1.525, 1.525], [-1.525, -1.525], [-20.0, 0.0], [-20.0, 0.0],
                      [-18.48, 1.525], [3.05, 1.525], [1.525, 0.0], [-20.0, 0.0], [-20.0, 0.0],
                      [-16.95, 1.525], [-3.05, -1.525], [-20.0, 0.0], [-23.05, -1.525]]


mu_cr_ltbeamn_mono = [
    46.24, 46.33, 35.87, 48.00, 10.92  
]

moments_ltbeamn_mono = [[0.0, 0.0], [1.388, 1.388], [-20.0, 0.0], [2.848, 1.388], [-17.15, 1.388]]


# ----------------------------------------------------------------------
def run_cases(section_type, node_sections, cases, mu_cr_ref):
    """Ejecuta una tanda de casos y retorna resultados."""
    print(f"\n{'─'*130}")
    print(f" {section_type} ".center(130))
    print(f"{'─'*130}")
    print(f"{'#':<3} {'Align':<6} {'Carga':<47} {'μ_cr (PyLTB)':>15} {'μ_cr (LTBeamN)':>15} {'Diff (%)':>10} {'M_i (kNm)':>12} {'M_j (kNm)':>12}")
    print(f"{'─'*130}")
    
    results = []
    for i, (align, fx, fz, fx_pos, fz_pos, desc) in enumerate(cases):
        nodal_loads = np.array([[nelems, fx_pos, fz_pos, 0.0, 0.0, fx, fz, 0.0]])

        model = StabilityModel()
        model.add_materials(materials)
        model.add_sections(node_sections)
        model.add_nodes(coordinates)
        model.add_tapered_elements(elements_data, align=align)
        model.add_verax_restraints(verax_restraints)
        model.add_lator_restraints(lator_restraints)
        model.add_nodal_loads(nodal_loads)

        static = StaticSolver(model)
        static.solve()

        M_i = -model.elements[0].forces[2] / 1e3
        M_j =  model.elements[-1].forces[5] / 1e3

        stabi = StabilitySolver(model)
        stabi.solve()
        mu_cr = stabi.mu_crs[0]

        mu_lt = mu_cr_ref[i] if i < len(mu_cr_ref) and mu_cr_ref[i] is not None else None
        diff = abs(mu_cr - mu_lt)/mu_lt*100 if mu_lt else None
        
        mu_lt_str = f"{mu_lt:.4f}" if mu_lt else "—"
        diff_str = f"{diff:.2f}" if diff else "—"
        
        print(f"{i:<3} {align:<6} {desc:<47} {mu_cr:>15.4f} {mu_lt_str:>15} {diff_str:>10} {M_i:>12.4f} {M_j:>12.4f}")
        
        results.append({
            'case': i, 'align': align, 'desc': desc,
            'mu_cr': mu_cr, 'mu_lt': mu_lt, 'diff': diff,
            'M_i': M_i, 'M_j': M_j
        })
    
    print(f"{'─'*130}")
    return results

# ----------------------------------------------------------------------
# Ejecutar ambas tandas
print("\n" + "="*130)
print(" TAPERED CANTILIVER – SYSTEMATIC TEST (TIP LOADS) ".center(130))
print("="*130)
print(" Longitud = 4 m, cargas axial y vertical puntuales".center(130))
print("="*130)
results_bi   = run_cases("BISYMMETRIC SECTIONS", node_sections_bi, cases_bi, mu_cr_ltbeamn_bi)
results_mono = run_cases("MONOSYMMETRIC SECTIONS", node_sections_mono, cases_mono, mu_cr_ltbeamn_mono)

# ----------------------------------------------------------------------