
"""
Test sistemático – Viga simplemente apoyada acartelada.
Cargas verticales puntuales hacia abajo, sin carga distribuida,
variando alineamiento, altura de carga, posición longitudinal y monosimetría.
"""

import numpy as np

from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# ----------------------------------------------------------------------
material1 = Material(E=2.10e11, nu=0.3, dens=1.0)
materials = [material1]

# Secciones bisimétricas
section1_bi = ISection_MS(h=0.61, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)
section2_bi = ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)

# Secciones monosimétricas (ala superior más ancha)
section1_mono = ISection_MS(h=0.61, bf1=0.24, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)
section2_mono = ISection_MS(h=0.305, bf1=0.24, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00)

L = 4.5
nelems = 36
nnods = nelems + 1
coordinates = np.linspace(0, L, nnods)
norm_coords = coordinates / L

node_sections_bi = interpolate_multiple_sections(section1_bi, section2_bi, norm_coords)
node_sections_mono = interpolate_multiple_sections(section1_mono, section2_mono, norm_coords)

elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])

# Apoyos: simplemente apoyada con horquillas (v=0, θ=0 en extremos)
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  0, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])

# ----------------------------------------------------------------------
mid_node = nelems // 2
Fz = -5.0e3
Fx = -10.0e3

# Casos bisimétricos (0-9)
cases_bi = [
    (0, 0.0,  Fz, 0, 0, "Fz en centroide"),
    (0, 0.0,  Fz, 0, 3, "Fz en ala superior"),
    (3, 0.0,  Fz, 0, 3, "Align 3, Fz en ala superior"),
    (3, 0.0,  Fz, 0, 0, "Align 3, Fz en centroide"),
    (0,  Fx, 0.0, 0, 0, "Fx en centroide"),
    (0,  Fx, 0.0, 3, 0, "Fx en ala superior (excéntrica)"),
    (3,  Fx, 0.0, 3, 0, "Align 3, Fx en ala superior"),
    (3,  Fx, 0.0, 0, 0, "Align 3, Fx en centroide"),
    (0,  Fx,  Fz, 3, 3, "Fx+Fz en ala superior"),
    (3,  Fx,  Fz, 3, 3, "Align 3, Fx+Fz en ala superior"),
]

# Casos monosimétricos (10-17)
cases_mono = [
    (0, 0.0,  Fz, 0, 0, "Fz en centroide"),
    (0, 0.0,  Fz, 0, 3, "Fz en ala superior"),
    (3, 0.0,  Fz, 0, 3, "Align 3, Fz en ala superior"),
    (0,  Fx, 0.0, 0, 0, "Fx en centroide"),
    (0,  Fx, 0.0, 3, 0, "Fx en ala superior"),
    (3,  Fx, 0.0, 3, 0, "Align 3, Fx en ala superior"),
    (0,  Fx,  Fz, 3, 3, "Fx+Fz en ala superior"),
    (3,  Fx,  Fz, 3, 3, "Align 3, Fx+Fz en ala superior"),
]

# ----------------------------------------------------------------------
# Referencias de LTBeamN (completar manualmente)
mu_cr_ltbeamn_bi = [
    61.75, 38.42, 38.41, 61.75, 188.57, 184.21, 166.98, 175.17, 33.97, 35.14
]

moments_ltbeamn_bi = [
    5.625, 5.625, 5.625, 5.625, 0.0, -1.144, -1.525, -0.3813, 6.769, 6.388
]

mu_cr_ltbeamn_mono = [
    127.59, 77.43, 77.39, 235.83, 229.96, 197.63, 68.89, 71.59
]

moments_ltbeamn_mono = [
    5.625, 5.625, 5.625, 0.0, 1.057, -1.424, 6.682, 6.314
]

# ----------------------------------------------------------------------
def run_cases(section_type, node_sections, cases, mu_cr_ref):
    print(f"\n{'─'*130}")
    print(f" {section_type} – SIMPLY SUPPORTED BEAM (load at midspan) ".center(130))
    print(f"{'─'*130}")
    print(f"{'#':<3} {'Align':<6} {'Carga':<57} {'μ_cr (PyLTB)':>15} {'μ_cr (LTBeamN)':>15} {'Diff (%)':>10} {'M_mid (kNm)':>13}")
    print(f"{'─'*130}")

    results = []
    for i, (align, fx, fz, fx_pos, fz_pos, desc) in enumerate(cases):
        nodal_loads = np.array([[mid_node, fx_pos, fz_pos, 0.0, 0.0, fx, fz, 0.0]])

        model = StabilityModel()
        model.add_materials(materials)
        model.add_sections(node_sections)
        model.add_nodes(coordinates)
        model.add_tapered_elements(elements_data, align=align)
        model.add_verax_restraints(verax_restraints)
        model.add_lator_restraints(lator_restraints)
        model.add_nodal_loads(nodal_loads)

        static = StaticSolver(model).solve()
        elem_mid = model.elements[mid_node - 1] if mid_node > 0 else model.elements[0]
        M_mid = elem_mid.forces[5] / 1e3

        stabi = StabilitySolver(model).solve()
        mu_cr = stabi.mu_crs[0]

        mu_lt = mu_cr_ref[i] if i < len(mu_cr_ref) and mu_cr_ref[i] is not None else None
        diff = abs(mu_cr - mu_lt)/mu_lt*100 if mu_lt else None

        mu_lt_str = f"{mu_lt:.4f}" if mu_lt else "—"
        diff_str = f"{diff:.2f}" if diff else "—"

        print(f"{i:<3} {align:<6} {desc:<57} {mu_cr:>15.4f} {mu_lt_str:>15} {diff_str:>10} {M_mid:>13.4f}")

        results.append({
            'case': i, 'align': align, 'desc': desc,
            'mu_cr': mu_cr, 'mu_lt': mu_lt, 'diff': diff,
            'M_mid': M_mid
        })

    print(f"{'─'*130}")
    return results

# ----------------------------------------------------------------------
print("\n" + "="*130)
print(" SIMPLY SUPPORTED TAPERED BEAM – SYSTEMATIC TEST (MIDSPAN LOADS) ".center(130))
print("="*130)
print(" Longitud = 4.5 m, cargas puntuales en el centro del vano".center(130))
print(" Apoyos de horquilla: v=0, θ=0 en ambos extremos".center(130))
print("="*130)

results_bi   = run_cases("BISYMMETRIC SECTIONS",   node_sections_bi,   cases_bi,   mu_cr_ltbeamn_bi)
results_mono = run_cases("MONOSYMMETRIC SECTIONS", node_sections_mono, cases_mono, mu_cr_ltbeamn_mono)
