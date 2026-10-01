
"""
Example 2  –  Cantilever tapered beam (L=4 m)
              Combined axial + transverse tip loads, varying N/Q ratio
"""
 
 
import numpy as np
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
 
 
# ── helpers ────────────────────────────────────────────────────────────────────
 
def solve(coords, sections, edata, nodal_loads, align=0):
    model = StabilityModel()
    model.add_materials([Material(E=2.10e11, nu=0.3, rho=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata, align=align)
    model.add_verax_restraints(np.array([[0, 1, 1, 1]]))
    model.add_lator_restraints(np.array([[0, 1, 1, 1, 1]]))
    model.add_nodal_loads(nodal_loads)
 
    s1 = StaticSolver(model);    s1.solve()
    s2 = StabilitySolver(model); s2.solve()
    return s1.max_vals(), s2.mu_crs[0]
 
 
def delta(ref, val):
    """Beyer (2015): Δ = (Ref - valor) / valor * 100."""
    return (ref - val) / val * 100


def print_row(label, mu, ref, ltb):
    print(f"  {label:>8}  {ref:>12.4f}  {ltb:>15.4f}  {delta(ref, ltb):>7.2f}%"
          f"  {mu:>14.4f}  {delta(ref, mu):>7.2f}%  {delta(ltb, mu):>10.2f}%")
 
 
# ── data ───────────────────────────────────────────────────────────────────────
 
L      = 4.0
nelems = 20
 
sec_i = ISection_MS(h=0.6127,       bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")
sec_j = ISection_MS(h=0.6127*0.2,   bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0, It_type="plates")
 
ratios   = [0, 1, 2, 4]
refs     = [1.979, 1.742, 1.475, 1.006]
ltbeamns = [1.944, 1.725, 1.472, 1.010] # programa
 
coords   = np.linspace(0, L, nelems + 1)
sections = interpolate_multiple_sections(sec_i, sec_j, coords / L)
edata    = np.array([[1, 0, e, e+1] for e in range(nelems)])
 
 
# ── run ────────────────────────────────────────────────────────────────────────
 
print("\n" + "═" * 92)
print("  Example 2  –  Cantilever tapered | combined N/Q tip loads  (L=4 m)")
print("═" * 92)
print(f"  {'r=N/Q':>8}  {'Reference':>12}  {'LTBeamN':>15}  {'Δ %':>8}"
      f"  {'PyLTB':>14}  {'Δ %':>8}  {'ΔLTB %':>11}")
print("  " + "─" * 88)
 
for r, ref, ltb in zip(ratios, refs, ltbeamns):
    loads = np.array([[nelems, 0, 3, 0.0, 0.0, r*-50000.0, -50000.0, 0.0]])
    _, mu = solve(coords, sections, edata, loads)
    print_row(f"r={r}", mu, ref, ltb)
 
print("\n" + "═" * 92 + "\n")
