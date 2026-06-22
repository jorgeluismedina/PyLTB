
"""
Example 1  –  Cantilever tapered beams
  1a: bisymmetric section,   Fz at top flange
  1b: monosymmetric section, Fz at top flange
  1c: bisymmetric section,   Fx + Fz at top flange (combined)
"""
 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import numpy as np
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
 
 
# ── helpers ────────────────────────────────────────────────────────────────────
 
def make_mesh(sec_i, sec_j, L, nelems):
    coords   = np.linspace(0, L, nelems + 1)
    sections = interpolate_multiple_sections(sec_i, sec_j, coords / L)
    edata    = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, sections, edata
 
 
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
 
 
def print_header(title, param_label):
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)
    print(f"  {param_label:>8}  {'μ_cr (PyLTB)':>14}  {'μ_cr (Ref)':>12}"
          f"  {'μ_cr (LTBeamN)':>15}  {'ΔRef %':>8}  {'ΔLTBeamN %':>8}")
    print("  " + "─" * 76)

def print_header2(title, param_label):
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)
    print(f"  {param_label:>8}  {'μ_cr (PyLTB)':>14}"
          f"  {'μ_cr (LTBeamN)':>15}  {'ΔLTBeamN %':>8}")
    print("  " + "─" * 76)
 
 
def print_row(label, mu, ref, ltb):
    dr = abs(mu - ref) / ref * 100
    dl = abs(mu - ltb) / ltb * 100
    print(f"  {label:>8}  {mu:>14.4f}  {ref:>12.4f}  {ltb:>15.4f}  {dr:>7.2f}%  {dl:>7.2f}%")
 
def print_row2(label, mu, ltb):
    dl = abs(mu - ltb) / ltb * 100
    print(f"  {label:>8}  {mu:>14.4f}  {ltb:>15.4f}  {dl:>7.2f}%")

# ── data ───────────────────────────────────────────────────────────────────────
 
Ls = [2, 4, 6, 8, 10]
 
sec_bs_i = ISection_MS(h=0.610, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0, r2=0)
sec_bs_j = ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0, r2=0)
 
sec_ms_i = ISection_MS(h=0.610, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0, r2=0)
sec_ms_j = ISection_MS(h=0.305, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0, r2=0)
 
 
# ── Example 1a ─────────────────────────────────────────────────────────────────
 
refs_1a     = [173.30, 44.55, 22.69, 13.95,  9.31]
ltbeamns_1a = [171.87, 44.23, 22.50, 13.82,  9.22]
 
print_header("Example 1a  –  bisymmetric tapered cantilever | Fz at top flange", "L")
for L, ref, ltb in zip(Ls, refs_1a, ltbeamns_1a):
    nelems = int(16 * L / 2)
    coords, sections, edata = make_mesh(sec_bs_i, sec_bs_j, L, nelems)
    loads = np.array([[nelems, 0, 3, 0.0, 0.0, 0.0, -1000.0, 0.0]])
    _, mu = solve(coords, sections, edata, loads, align=3)
    print_row(f"{L} m", mu, ref, ltb)
 
 
# ── Example 1b ─────────────────────────────────────────────────────────────────
 
refs_1b     = [77.21, 26.76, 15.08,  9.680,  6.610]
ltbeamns_1b = [75.54, 26.55, 14.93,  9.572,  6.537]
 
print_header("Example 1b  –  monosymmetric tapered cantilever | Fz at top flange", "L")
for L, ref, ltb in zip(Ls, refs_1b, ltbeamns_1b):
    nelems = int(16 * L / 2)
    coords, sections, edata = make_mesh(sec_ms_i, sec_ms_j, L, nelems)
    loads = np.array([[nelems, 3, 3, 0.0, 0.0, 0.0, -1000.0, 0.0]])
    _, mu = solve(coords, sections, edata, loads, align=3)
    print_row(f"{L} m", mu, ref, ltb)
 
 
# ── Example 1c ─────────────────────────────────────────────────────────────────
 
ltbeamns_1c = [149.8, 40.75, 21.01, 12.95, 8.646]
 
print_header2("Example 1c  –  bisymmetric tapered cantilever | Fx + Fz at top flange", "L")
for L, ltb in zip(Ls, ltbeamns_1c):
    nelems = int(16 * L / 2)
    coords, sections, edata = make_mesh(sec_bs_i, sec_bs_j, L, nelems)
    loads = np.array([[nelems, 3, 3, 0.0, 0.0, -1000.0, -1000.0, 0.0]])
    _, mu = solve(coords, sections, edata, loads, align=3)
    print_row2(f"{L} m", mu, ltb)
 
print("\n" + "═" * 78 + "\n")