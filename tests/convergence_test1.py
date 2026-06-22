
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
 
# ── Material & sección ────────────────────────────────────────────────────────
material = Material(E=2.1e11, nu=0.3, dens=1.0)
 
section = ISection_MS(h=0.3, bf1=0.15, bf2=0.15,
                      tw=0.015, tf1=0.015, tf2=0.015,
                      r1=0.01, r2=0.01)
 
L = 5.0   # [m]
M0 = 1000.0  # [Nm] momento de flexión pura
 
# ── Referencia analítica ──────────────────────────────────────────────────────
EIz = material.E * section.Iz
GIt = material.G * section.It
EIw = material.E * section.Iw
mu_cr_ana = np.pi / L * np.sqrt(EIz * GIt * (1 + np.pi**2 * EIw / (L**2 * GIt))) / M0
 
print(f"Analytical μ_cr = {mu_cr_ana:.6f}\n")
 
 
# ── Función principal ─────────────────────────────────────────────────────────
def run_model(nelems, etype):
    """
    etype 0 → LTBeam    (matrices cerradas, uniforme)
    etype 1 → LTBeamTap (integración de Gauss, tapered)
    """
    coordinates = np.linspace(0, L, nelems + 1)
    node_sections = [section] * (nelems + 1)
 
    elements_data = np.array([
        [etype, 0, e, e + 1] for e in range(nelems)
    ])
 
    verax_restraints = np.array([
        [0,      1, 1, 0],
        [nelems, 0, 1, 0],
    ])
    lator_restraints = np.array([
        [0,      1, 0, 1, 0],
        [nelems, 1, 0, 1, 0],
    ])
    # [node, pos, Px, Pz, M]
    nodal_loads = np.array([
        [0,      0, 1,  0.0, 0.0,  0.0, 0.0, -M0],
        [nelems, 0, 1,  0.0, 0.0,  0.0, 0.0,  M0],
    ])
 
    model = StabilityModel()
    model.add_materials([material])
    model.add_sections(node_sections)
    model.add_nodes(coordinates)
 
    if etype == 0:
        model.add_uniform_elements(elements_data)
    else:
        model.add_tapered_elements(elements_data)
 
    model.add_verax_restraints(verax_restraints)
    model.add_lator_restraints(lator_restraints)
    model.add_nodal_loads(nodal_loads)
 
    StaticSolver(model).solve()
 
    stab = StabilitySolver(model)
    stab.solve()
    return stab.mu_crs[0]
 
 
# ── Estudio de convergencia ───────────────────────────────────────────────────
mesh_sizes = [2, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100]
#mesh_sizes = [2, 4, 6, 8, 10, 15, 20]
 
mu_uni, mu_tap = [], []
 
print(f"{'n':>6}  {'μ_uni':>12}  {'err_uni %':>10}  {'μ_tap':>12}  {'err_tap %':>10}")
print("-" * 60)
 
for n in mesh_sizes:
    mu_u = run_model(n, etype=0)
    mu_t = run_model(n, etype=1)
    mu_uni.append(mu_u)
    mu_tap.append(mu_t)
    err_u = abs(mu_u - mu_cr_ana) / mu_cr_ana * 100
    err_t = abs(mu_t - mu_cr_ana) / mu_cr_ana * 100
    print(f"{n:>6}  {mu_u:>12.6f}  {err_u:>10.4f}  {mu_t:>12.6f}  {err_t:>10.4f}")
 
mu_uni = np.array(mu_uni)
mu_tap = np.array(mu_tap)
err_uni = np.abs(mu_uni - mu_cr_ana) / mu_cr_ana * 100
err_tap = np.abs(mu_tap - mu_cr_ana) / mu_cr_ana * 100
ns = np.array(mesh_sizes)
 
# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#fig.suptitle(
#    r"Convergence Study – Simply Supported Beam, Pure Bending"
#    "\n"
#    r"$\mu_{cr}^{\mathrm{ref}} = $" + f"{mu_cr_ana:.4f}",
#    fontsize=12,
#)
 
# μ_cr vs n
ax = axes[0]
ax.axhline(mu_cr_ana, color="k", ls="--", lw=1.2, label=r"Analytical $\mu_{cr}$")
ax.plot(ns, mu_uni, "o-", color="steelblue",  lw=1.5, ms=6, label="Uniform")
ax.plot(ns, mu_tap, "s-", color="darkorange", lw=1.5, ms=6, label="Tapered")
ax.set_xlabel("Number of elements  $n$", fontsize=11)
ax.set_ylabel(r"$\mu_{cr}$", fontsize=11)
ax.set_title(r"Critical load factor vs mesh refinement")
ax.set_xscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
 
# Error relativo vs n
ax = axes[1]
ax.plot(ns, err_uni, "o-", color="steelblue",  lw=1.5, ms=6, label="Uniform")
ax.plot(ns, err_tap, "s-", color="darkorange", lw=1.5, ms=6, label="Tapered")
ax.set_xlabel("Number of elements  $n$", fontsize=11)
ax.set_ylabel(r"Error  [%]", fontsize=11)
ax.set_title("Relative error vs mesh refinement")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")
 
plt.tight_layout()
plt.savefig("convergence_plot1.pdf", dpi=200)
plt.show()
print("\nDone.")