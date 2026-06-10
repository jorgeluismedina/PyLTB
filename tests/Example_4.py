
"""
Example 4  –  Simply supported monosymmetric double-tapered beam
              Mid-span point load at top flange (load applied at shear-center line)
  full:  complete beam, 3 lengths
  sym:   half-beam with symmetric BCs, 3 lengths
"""
 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import numpy as np
from src.model import StabilityModel
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.sections.section_utils import interpolate_multiple_sections
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
 
 
# ── helpers ────────────────────────────────────────────────────────────────────
 
def make_mesh_full(sec_min, sec_max, L, nelems):
    nnods  = nelems + 1
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
    model.add_materials([Material(E=2.10e11, nu=0.3, dens=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata)
    model.add_verax_restraints(vrest)
    model.add_lator_restraints(lrest)
    model.add_nodal_loads(nodal_loads)
 
    s1 = StaticSolver(model);    s1.solve()
    s2 = StabilitySolver(model); s2.solve()
    return s1.max_vals(), s2.mu_crs[0]
 
 
def print_header(title):
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)
    print(f"  {'L':>6}  {'μ_cr (PyLTB)':>14}  {'μ_cr (Ref)':>12}"
          f"  {'μ_cr (LTBeamN)':>15}  {'ΔRef %':>8}  {'ΔLTBeamN %':>8}")
    print("  " + "─" * 76)
 
 
def print_row(label, mu, ref, ltb):
    dr = abs(mu - ref) / ref * 100
    dl = abs(mu - ltb) / ltb * 100
    print(f"  {label:>6}  {mu:>14.4f}  {ref:>12.4f}  {ltb:>15.4f}  {dr:>7.2f}%  {dl:>7.2f}%")
 
 
# ── data ───────────────────────────────────────────────────────────────────────
 
sec_max = ISection_MS(h=0.60,      bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0)
sec_min = ISection_MS(h=0.60*0.4,  bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0, r2=0)
 
# Eccentricity correction: offset between SC at support (min) and SC at midspan (max)
rez = np.abs(sec_min.z_from_ref(0, 1) - sec_max.z_from_ref(0, 1))
 
Ls       = [6, 9, 12]
refs     = [85.09, 39.27, 22.47]
ltbeamns = [83.73, 38.71, 22.18]
 
 
# ── Example 4 – full ───────────────────────────────────────────────────────────
 
print_header("Example 4 (full)  –  S-S monosymmetric double-taper | mid-span Fz at top flange")
for L, ref, ltb in zip(Ls, refs, ltbeamns):
    nelems = int(10 * L / 2)
    coords, sections, edata = make_mesh_full(sec_min, sec_max, L, nelems)
    vrest = np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]])
    loads = np.array([[nelems//2, 0, 3, 0.0, -rez, 0.0, -1000.0, 0.0]])
    _, mu = solve(coords, sections, edata, vrest, lrest, loads)
    print_row(f"{L} m", mu, ref, ltb)
 
 
# ── Example 4 – symmetric (half model) ────────────────────────────────────────
 
ltbeamns_sym = [83.97, 38.64, 22.09]
 
print_header("Example 4 (sym)   –  half-model with symmetric BCs")
for L, ref, ltb in zip(Ls, refs, ltbeamns_sym):
    L_half = L / 2
    nelems = int(10 * L_half)
    coords, sections, edata = make_mesh_half(sec_min, sec_max, L_half, nelems)
    vrest = np.array([[0, 0, 1, 0], [nelems, 1, 0, 1]])
    lrest = np.array([[0, 1, 0, 1, 0], [nelems, 0, 1, 0, 1]])
    loads = np.array([[nelems, 0, 3, 0.0, -rez, 0.0, -500.0, 0.0]])
    _, mu = solve(coords, sections, edata, vrest, lrest, loads)
    print_row(f"{L} m", mu, ref, ltb)
 
print("\n" + "═" * 80 + "\n")