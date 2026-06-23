
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
    model.add_materials([Material(E=2.10e11, nu=0.3, dens=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata, align=align)
    model.add_verax_restraints(np.array([[0, 1, 1, 1]]))
    model.add_lator_restraints(np.array([[0, 1, 1, 1, 1]]))
    model.add_nodal_loads(nodal_loads)
 
    s1 = StaticSolver(model);    s1.solve()
    s2 = StabilitySolver(model); s2.solve()
    return s1.max_vals(), s2.mu_crs[0]
 
 
def print_row(label, mu, ref, ltb):
    dr = abs(mu - ref) / ref * 100
    dl = abs(mu - ltb) / ltb * 100
    print(f"  {label:>8}  {mu:>14.4f}  {ref:>12.4f}  {ltb:>15.4f}  {dr:>7.2f}%  {dl:>7.2f}%")
 
 
# ── data ───────────────────────────────────────────────────────────────────────
 
L      = 4.0
nelems = 20
 
sec_i = ISection_MS(h=0.6127,       bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0)
sec_j = ISection_MS(h=0.6127*0.2,   bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0)
 
ratios   = [0, 1, 2, 4]
refs     = [1.979, 1.742, 1.475, 1.006]
ltbeamns = [1.943, 1.722, 1.469, 1.010]
 
coords   = np.linspace(0, L, nelems + 1)
sections = interpolate_multiple_sections(sec_i, sec_j, coords / L)
edata    = np.array([[1, 0, e, e+1] for e in range(nelems)])
 
 
# ── run ────────────────────────────────────────────────────────────────────────
 
print("\n" + "═" * 80)
print("  Example 2  –  Cantilever tapered | combined N/Q tip loads  (L=4 m)")
print("═" * 80)
print(f"  {'r=N/Q':>8}  {'μ_cr (PyLTB)':>14}  {'μ_cr (Ref)':>12}"
      f"  {'μ_cr (LTBeamN)':>15}  {'ΔRef %':>8}  {'ΔLTBeamN %':>8}")
print("  " + "─" * 76)
 
for r, ref, ltb in zip(ratios, refs, ltbeamns):
    loads = np.array([[nelems, 0, 3, 0.0, 0.0, r*-50000.0, -50000.0, 0.0]])
    _, mu = solve(coords, sections, edata, loads)
    print_row(f"r={r}", mu, ref, ltb)
 
print("\n" + "═" * 80 + "\n")
