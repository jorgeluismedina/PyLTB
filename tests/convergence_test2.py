
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
 
# ── Material & sección ────────────────────────────────────────────────────────
material = Material(E=2.1e11, nu=0.3, rho=1.0)
 
section = ISection_MS(h=0.3, bf1=0.15, bf2=0.15,
                      tw=0.015, tf1=0.015, tf2=0.015,
                      r1=0.01, r2=0.01)
 
L = 5.0   # [m]
qz = -1000.0  # [N/m] carga vertical hacia abajo
 
# ── Referencia LTBeamN ──────────────────────────────────────────────────────
mu_cr_ltbeamn = 82.41
print(f"LTBeamN μ_cr = {mu_cr_ltbeamn:.6f}\n")
 
 
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

    elem_loads = np.array([
        [e, 0, 1, 0.0, 0.0,   0.0, -1000.0, 0.0, -1000.0] for e in range(nelems)
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
    model.add_elem_loads(elem_loads)
 
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
    err_u = abs(mu_u - mu_cr_ltbeamn) / mu_cr_ltbeamn * 100
    err_t = abs(mu_t - mu_cr_ltbeamn) / mu_cr_ltbeamn * 100
    print(f"{n:>6}  {mu_u:>12.6f}  {err_u:>10.4f}  {mu_t:>12.6f}  {err_t:>10.4f}")
 
mu_uni = np.array(mu_uni)
mu_tap = np.array(mu_tap)
err_uni = np.abs(mu_uni - mu_cr_ltbeamn) / mu_cr_ltbeamn * 100
err_tap = np.abs(mu_tap - mu_cr_ltbeamn) / mu_cr_ltbeamn * 100
ns = np.array(mesh_sizes)
 
# ── Plots ─────────────────────────────────────────────────────────────────────
plt.style.use(['science','notebook','grid'])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#fig.suptitle(
#    r"Convergence Study – Simply Supported Beam, Uniform Distributed V. Load"
#    "\n"
#    r"$\mu_{cr}^{\mathrm{ref}} = $" + f"{mu_cr_ltbeamn:.4f}",
#    fontsize=12,
#)
fs_axes = 15
fs_ticks = 13
fs_legend = 12
ms_plot = 5

# μ_cr vs n
ax = axes[0]
ax.axhline(mu_cr_ltbeamn, color="k", ls="--", lw=1.2, label=r"LTBeamN $\mu_{cr}$")
ax.plot(ns, mu_uni, "o-", color="blue",  lw=1.5, ms=6, label="Uniform element type")
ax.plot(ns, mu_tap, "s-", color="red", lw=1.5, ms=6, label="Tapered element type")
ax.set_xlabel("Number of elements  $n$", fontsize=fs_axes)
ax.set_ylabel(r"$\mu_{cr}$", fontsize=fs_axes)
#ax.set_title(r"Critical load factor vs mesh refinement")
ax.set_xscale("log")
ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fancybox=False, edgecolor='black', fontsize=fs_legend)
ax.grid(True, alpha=0.3)
 
# Error relativo vs n
ax = axes[1]
ax.plot(ns, err_uni, "o-", color="blue",  lw=1.5, ms=6, label="Uniform element type")
ax.plot(ns, err_tap, "s-", color="red", lw=1.5, ms=6, label="Tapered element type")
ax.set_xlabel("Number of elements  $n$", fontsize=fs_axes)
ax.set_ylabel(r"Error  [%]", fontsize=fs_axes)
#ax.set_title("Relative error vs mesh refinement")
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fancybox=False, edgecolor='black', fontsize=fs_legend)
ax.grid(True, alpha=0.3, which="both")
 
plt.tight_layout()
plt.savefig(r"D:\Maestria UFRGS\Tesis maestria\Seminario\fig\convergence_plot2.pdf", dpi=300)
plt.show()
print("\nDone.")
