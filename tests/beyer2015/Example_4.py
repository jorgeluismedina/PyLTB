"""
Example 4  –  Beyer et al. (2015), §4.5
              Simply supported monosymmetric double-tapered beam, sections aligned at their
              centroids (the shear center line has a kink at mid-span).
              Mid-span point load at the top flange.
  full:  complete beam, 3 lengths
  sym:   half-beam with symmetric BCs, 3 lengths

Two ways of measuring the load height e_z (as in Beyer 2015, Table 5):
  e(SC): from the local shear center (what the program does by default)
  e(TC): from the torsion center line, the straight line through the shear centers of the
         support sections. Emulated by lowering the load point by rez from the top flange.

Δ = (Ref - value) / value * 100, the definition of Beyer (2015).
"""


import numpy as np
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── helpers ────────────────────────────────────────────────────────────────────

def make_mesh_full(sec_min, sec_max, L, nelems):
    nnods  = nelems + 1                      # nelems par: el nodo nelems//2 está en el centro
    coords = np.linspace(0, L, nnods)
    norm   = coords / L
    half   = nnods // 2
    secs   = (interpolate_multiple_sections(sec_min, sec_max, norm[:half+1] * 2.0) +
              interpolate_multiple_sections(sec_max, sec_min, (norm[half+1:] - 0.5) * 2.0))
    edata  = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, secs, edata


def make_mesh_half(sec_min, sec_max, L_half, nelems):
    coords   = np.linspace(0, L_half, nelems + 1)
    sections = interpolate_multiple_sections(sec_min, sec_max, coords / L_half)
    edata    = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, sections, edata


def solve(coords, sections, edata, vrest, lrest, nodal_loads):
    model = StabilityModel()
    model.add_materials([Material(E=2.10e11, nu=0.3, rho=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata)
    model.add_verax_restraints(vrest)
    model.add_lator_restraints(lrest)
    model.add_nodal_loads(nodal_loads)

    StaticSolver(model).solve()
    return StabilitySolver(model).solve().mu_crs[0]


def delta(ref, val):
    return (ref - val) / val * 100


def print_header(title):
    w = 118
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)
    print(f"  {'':>5}  {'':>9}  {'LTBeamN (programa)':^40}  {'PyLTB':^53}")
    print(f"  {'L [m]':>5}  {'Ref.':>9}  {'e(TC)':>9} {'Δ %':>8}  {'e(SC)':>9} {'Δ %':>8}"
          f"   {'e(TC)':>9} {'Δ %':>8} {'ΔLTB %':>8}  {'e(SC)':>9} {'Δ %':>8} {'ΔLTB %':>8}")
    print("  " + "─" * (w - 2))


def print_row(L, ref, ltb_tc, ltb_sc, mu_tc, mu_sc):
    print(f"  {L:>5}  {ref:>9.2f}  {ltb_tc:>9.2f} {delta(ref, ltb_tc):>7.2f}%"
          f"  {ltb_sc:>9.2f} {delta(ref, ltb_sc):>7.2f}%"
          f"   {mu_tc:>9.2f} {delta(ref, mu_tc):>7.2f}% {delta(ltb_tc, mu_tc):>7.2f}%"
          f"  {mu_sc:>9.2f} {delta(ref, mu_sc):>7.2f}% {delta(ltb_sc, mu_sc):>7.2f}%")


# ── data ───────────────────────────────────────────────────────────────────────

sec_max = ISection_MS(h=0.60,      bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")
sec_min = ISection_MS(h=0.60*0.4,  bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")

# Distancia entre el SC de mitad de luz y la línea de centros de torsión (SC de los apoyos),
# medida con el eje de centroides recto (align = 0)
rez = np.abs(sec_min.z_from_ref(0, 1) - sec_max.z_from_ref(0, 1))

Ls          = [6, 9, 12]
refs        = [85.09, 39.27, 22.47]   # Ansys (Beyer 2015, Table 5)
ltbeamns_TC = [80.56, 37.52, 21.60]   # LTBeamN, programa
ltbeamns_SC = [56.20, 28.99, 17.69]   # LTBeamN, programa
# Valores del artículo (Beyer 2015, Table 5), que no coinciden con los del programa:
#   e(TC) = [83.73, 38.71, 22.18],  e(SC) = [57.94, 29.71, 18.14]

# Carga en la mesa superior: [fzpos, fzez] para cada forma de medir e_z
load_pos = {"TC": (3, -rez),   # bajar rez desde la mesa superior
            "SC": (3, 0.0)}
dens = 10                         # elementos por metro


# ── Example 4 – full ───────────────────────────────────────────────────────────

print_header("Example 4 (full)  –  S-S monosymmetric double-taper | mid-span Fz at top flange")
for L, ref, ltb_tc, ltb_sc in zip(Ls, refs, ltbeamns_TC, ltbeamns_SC):
    nelems = dens * L
    coords, sections, edata = make_mesh_full(sec_min, sec_max, L, nelems)
    vrest = np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]])
    mu = {k: solve(coords, sections, edata, vrest, lrest,
                   np.array([[nelems//2, 0, p, 0.0, ez, 0.0, -1000.0, 0.0]]))
          for k, (p, ez) in load_pos.items()}
    print_row(L, ref, ltb_tc, ltb_sc, mu["TC"], mu["SC"])


# ── Example 4 – symmetric (half model) ────────────────────────────────────────

print_header("Example 4 (sym)   –  half-model with symmetric BCs")
for L, ref, ltb_tc, ltb_sc in zip(Ls, refs, ltbeamns_TC, ltbeamns_SC):
    L_half = L / 2
    nelems = int(dens * L_half)
    coords, sections, edata = make_mesh_half(sec_min, sec_max, L_half, nelems)
    vrest = np.array([[0, 0, 1, 0], [nelems, 1, 0, 1]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 0, 1, 0, 1]])
    mu = {k: solve(coords, sections, edata, vrest, lrest,
                   np.array([[nelems, 0, p, 0.0, ez, 0.0, -500.0, 0.0]]))
          for k, (p, ez) in load_pos.items()}
    print_row(L, ref, ltb_tc, ltb_sc, mu["TC"], mu["SC"])

print("\n  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100, con la misma forma de medir e_z")
print("═" * 118 + "\n")
