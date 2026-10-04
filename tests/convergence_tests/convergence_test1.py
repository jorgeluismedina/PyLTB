"""
Test de convergencia: viga simplemente apoyada en flexión pura
Sección I doblemente simétrica, momentos iguales y opuestos M0 en los extremos (momento uniforme).
Apoyos de horquilla: v = θ = 0 en ambos extremos; v_x y θ_x libres (alabeo libre).

Solución exacta (Timoshenko y Gere 1961, cap. 6):
    M_cr = (π/L) √( EI_z GI_t (1 + π² EI_w / (GI_t L²)) ),   μ_cr = M_cr / M0
"""
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
                      r1=0.0, r2=0.0)
 
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
    print(f"{n:>6}  {mu_u:>12.6f}  {err_u:>10.2e}  {mu_t:>12.6f}  {err_t:>10.2e}")
 
mu_uni = np.array(mu_uni)
mu_tap = np.array(mu_tap)
err_uni = np.abs(mu_uni - mu_cr_ana) / mu_cr_ana * 100
err_tap = np.abs(mu_tap - mu_cr_ana) / mu_cr_ana * 100
ns = np.array(mesh_sizes)
 
# ── Plots ─────────────────────────────────────────────────────────────────────
plt.style.use(["science", "notebook", "grid",
               {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True}])
fs_axes, fs_ticks, fs_legend = 16, 14, 12
n_ticks = [2, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# μ_cr vs n
ax = axes[0]
ax.axhline(mu_cr_ana, color="k", ls="--", lw=1.2, label=r"Analytical $\mu_{cr}$")
ax.plot(ns, mu_uni, "o-", color="blue", lw=1.5, ms=6, label="Uniform element type")
ax.plot(ns, mu_tap, "s--", color="red", lw=1.5, ms=8, mfc="none", mew=1.2, label="Tapered element type")
ax.set_ylabel(r"$\mu_{cr}$", fontsize=fs_axes)

# Error relativo vs n
ax = axes[1]
ax.plot(ns, err_uni, "o-", color="blue", lw=1.5, ms=6, label="Uniform element type")
ax.plot(ns, err_tap, "s--", color="red", lw=1.5, ms=8, mfc="none", mew=1.2, label="Tapered element type")
ax.set_ylabel("Error [%]", fontsize=fs_axes)
ax.set_yscale("log")

for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(n_ticks, labels=[str(n) for n in n_ticks])
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel("Number of elements $n$", fontsize=fs_axes)
    ax.tick_params(axis="both", which="major", labelsize=fs_ticks)
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95), fancybox=False, edgecolor="black",
              fontsize=fs_legend)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig(r"D:\Maestria UFRGS\Tesis maestria\Disertacion\fig\convergence_plot1.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\nDone.")
