
"""
Example 3  –  Simply supported bisymmetric double-tapered beam
              Mid-span point load at top flange
  full:  complete beam, 3 lengths
  sym:   half-beam with symmetric BCs, 3 lengths

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
    """Double-taper: min → max → min over the full span."""
    nnods  = nelems + 1                      # nelems par: el nodo nelems//2 está en el centro
    coords = np.linspace(0, L, nnods)
    norm   = coords / L
    half   = nnods // 2
    secs   = (interpolate_multiple_sections(sec_min, sec_max, norm[:half+1] * 2.0) +
              interpolate_multiple_sections(sec_max, sec_min, (norm[half+1:] - 0.5) * 2.0))
    edata  = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, secs, edata
 
 
def make_mesh_half(sec_min, sec_max, L_half, nelems):
    """Half-beam: min → max."""
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
 
    s1 = StaticSolver(model);    s1.solve()
    s2 = StabilitySolver(model); s2.solve()
    return s1.max_vals(), s2.mu_crs[0]
 
 
def delta(ref, val):
    return (ref - val) / val * 100


def print_header(title):
    print("\n" + "═" * 92)
    print(f"  {title}")
    print("═" * 92)
    print(f"  {'L [m]':>8}  {'Reference':>12}  {'LTBeamN':>15}  {'Δ %':>8}"
          f"  {'PyLTB':>14}  {'Δ %':>8}  {'ΔLTB %':>11}")
    print("  " + "─" * 88)


def print_row(label, mu, ref, ltb):
    print(f"  {label:>8}  {ref:>12.4f}  {ltb:>15.4f}  {delta(ref, ltb):>7.2f}%"
          f"  {mu:>14.4f}  {delta(ref, mu):>7.2f}%  {delta(ltb, mu):>10.2f}%")


# ── data ───────────────────────────────────────────────────────────────────────
 
sec_max = ISection_MS(h=0.60,      bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")
sec_min = ISection_MS(h=0.60*0.4,  bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")
 
Ls       = [6, 9, 12]
refs     = [63.58, 30.55, 18.05]   # Ansys (Beyer 2015)
ltbeamns = [61.20, 29.65, 17.60]   # LTBeamN, programa
dens     = 10                      # elementos por metro
 
# ── Example 3 – full ───────────────────────────────────────────────────────────
 
print_header("Example 3 (full)  –  S-S bisymmetric double-taper | mid-span Fz at top flange")
for L, ref, ltb in zip(Ls, refs, ltbeamns):
    nelems = dens * L
    coords, sections, edata = make_mesh_full(sec_min, sec_max, L, nelems)
    vrest = np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]])
    loads = np.array([[nelems//2, 0, 3, 0.0, 0.0, 0.0, -1000.0, 0.0]])
    _, mu = solve(coords, sections, edata, vrest, lrest, loads)
    print_row(f"{L}", mu, ref, ltb)
 
 
# ── Example 3 – symmetric (half model) ────────────────────────────────────────

 
print_header("Example 3 (sym)   –  half-model with symmetric BCs")
for L, ref, ltb in zip(Ls, refs, ltbeamns):
    L_half = L / 2
    nelems = int(dens * L_half)
    coords, sections, edata = make_mesh_half(sec_min, sec_max, L_half, nelems)
    vrest = np.array([[0, 0, 1, 0], [nelems, 1, 0, 1]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 0, 1, 0, 1]])
    loads = np.array([[nelems, 0, 3, 0.0, 0.0, 0.0, -500.0, 0.0]])
    _, mu = solve(coords, sections, edata, vrest, lrest, loads)
    print_row(f"{L}", mu, ref, ltb)
 
print("\n  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100")
print("═" * 92 + "\n")
