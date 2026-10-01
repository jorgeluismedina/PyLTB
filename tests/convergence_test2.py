"""
Test de convergencia: ménsula de Prandtl y Michell (1899)
Sección rectangular angosta (sin alas → I_w = 0), carga puntual P en el extremo libre,
aplicada en el centroide (= centro de corte).

Solución exacta (Timoshenko y Gere 1961, cap. 6):
    P_cr = γ √(EI_z GI_t) / L²,   con γ/2 la primera raíz de J_{-1/4}  →  γ = 4.0126
Con I_w = 0 la ecuación de torsión es de 2.º orden: θ = 0 en el empotramiento y θ_x libre
(detalle en claude/output/mensula_prandtl.md).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import brentq
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# ── Material & sección ────────────────────────────────────────────────────────
material = Material(E=2.1e11, nu=0.3, rho=1.0)

# Rectángulo angosto 300 x 15 mm: alas nulas (bf = tf = 0), sin fillets
section = ISection_MS(h=0.3, bf1=0.0, bf2=0.0,
                      tw=0.015, tf1=0.0, tf2=0.0,
                      r1=0.0, r2=0.0, It_type="plates")

L = 5.0      # [m]
P = 1000.0   # [N] carga en el extremo libre, hacia abajo

# ── Referencia analítica ──────────────────────────────────────────────────────
gamma = 2 * brentq(lambda x: jv(-0.25, x), 1.5, 2.5)   # J_{-1/4}(γ/2) = 0
EIz = material.E * section.Iz
GIt = material.G * section.It
mu_cr_ana = gamma * np.sqrt(EIz * GIt) / L**2 / P

print(f"I_w = {section.Iw:.3e},  z_S = {section.zS:.3e},  β_z = {section.beta_z:.3e}")
print(f"γ = {gamma:.10f}")
print(f"Analytical μ_cr = {mu_cr_ana:.6f}\n")


# ── Función principal ─────────────────────────────────────────────────────────
def run_model(nelems, etype):
    """
    etype 0 → BeamP  (matrices cerradas, uniforme)
    etype 1 → BeamNP (integración de Gauss, tapered)
    """
    coordinates = np.linspace(0, L, nelems + 1)
    node_sections = [section] * (nelems + 1)

    elements_data = np.array([
        [etype, 0, e, e + 1] for e in range(nelems)
    ])

    # Empotramiento en el nodo 0: [node, u, w, w_x]
    verax_restraints = np.array([
        [0, 1, 1, 1],
    ])
    # [node, v, v_x, θ, θ_x]: θ_x libre, con I_w = 0 no hay condición de alabeo
    lator_restraints = np.array([
        [0, 1, 1, 1, 0],
    ])
    # [node, fxpos, fzpos, fxez, fzez, Fx, Fz, Mx]: Fz en el centroide del extremo libre
    nodal_loads = np.array([
        [nelems, 0, 0, 0.0, 0.0, 0.0, -P, 0.0],
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
plt.style.use(['science','notebook','grid'])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fs_axes = 15
fs_ticks = 13
fs_legend = 12

# μ_cr vs n
ax = axes[0]
ax.axhline(mu_cr_ana, color="k", ls="--", lw=1.2, label=r"Analytical $\mu_{cr}$")
ax.plot(ns, mu_uni, "o-", color="blue",  lw=1.5, ms=6, label="Uniform element type")
ax.plot(ns, mu_tap, "s-", color="red", lw=1.5, ms=6, label="Tapered element type")
ax.set_xlabel("Number of elements  $n$", fontsize=fs_axes)
ax.set_ylabel(r"$\mu_{cr}$", fontsize=fs_axes)
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
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fancybox=False, edgecolor='black', fontsize=fs_legend)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(r"D:\Maestria UFRGS\Tesis maestria\Disertacion\fig\convergence_plot2.pdf", dpi=300)
plt.show()
print("\nDone.")
