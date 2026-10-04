"""
Test de convergencia: voladizo acartelado, carga puntual P en el extremo libre sobre el ala superior
Geometría del ejemplo 1 de Beyer et al. (2015) con L = 6 m: canto lineal de 610 a 305 mm, alas
constantes de 10 mm, alma de 8 mm. El ala superior es recta (align=3).
  1a: alas de 180 mm (bisimétrica)          → integrandos polinómicos en x; ejes de C y S rectos
  1b: alas de 100 y 180 mm (monosimétrica)  → z_C y β_z no son polinomios en x; eje de C curvo,
      eje de S recto (z_S solo de las alas, constantes)

Sin solución analítica. Se separan los dos errores numéricos de los elementos BeamNP:
  1. Cuadratura: 4 elementos y reglas de Gauss-Legendre de 2 a 8 puntos
     (numpy.polynomial.legendre.leggauss). Referencia: la regla de 12 puntos.
  2. Discretización: la regla de 4 puntos de PyLTB y de 2 a 100 elementos.
     Referencia: 200 elementos.
"""
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# ── Material & secciones ──────────────────────────────────────────────────────
material = Material(E=2.1e11, nu=0.3, rho=1.0)

# (empotramiento, extremo libre)
sections = {
    "1a": (ISection_MS(h=0.610, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010,
                       r1=0.0, r2=0.0, It_type="plates"),
           ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010,
                       r1=0.0, r2=0.0, It_type="plates")),
    "1b": (ISection_MS(h=0.610, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010,
                       r1=0.0, r2=0.0, It_type="plates"),
           ISection_MS(h=0.305, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010,
                       r1=0.0, r2=0.0, It_type="plates")),
}

L = 6.0      # [m]
P = 1000.0   # [N] carga en el extremo libre, hacia abajo


# ── Funciones ─────────────────────────────────────────────────────────────────
def set_gauss_rule(model, npts):
    """Cambia la regla de Gauss de los elementos BeamNP (PyLTB usa 4 puntos) y rearma su K0."""
    x, w = np.polynomial.legendre.leggauss(npts)
    for elem in model.elements:
        elem.gpoints, elem.gweights = (x + 1) / 2, w / 2      # de [-1, 1] a [0, 1]
        elem.K0_vrx, elem.K0_ltr = np.zeros((6, 6)), np.zeros((8, 8))
        elem.compute_K0_matrices()


def run_model(case, nelems, npts=4):
    """μ_cr del voladizo con nelems elementos BeamNP y una regla de Gauss de npts puntos."""
    sec_root, sec_tip = sections[case]
    coordinates = np.linspace(0, L, nelems + 1)
    node_sections = interpolate_multiple_sections(sec_root, sec_tip, coordinates / L)

    elements_data = np.array([
        [1, 0, e, e + 1] for e in range(nelems)
    ])

    # Empotramiento
    verax_restraints = np.array([[0, 1, 1, 1]])
    lator_restraints = np.array([[0, 1, 1, 1, 1]])
    # Carga P sobre el ala superior en el extremo libre
    nodal_loads = np.array([
        [nelems, 0, 3,  0.0, 0.0,  0.0, -P, 0.0],
    ])

    model = StabilityModel()
    model.add_materials([material])
    model.add_sections(node_sections)
    model.add_nodes(coordinates)
    model.add_tapered_elements(elements_data, align=3)
    set_gauss_rule(model, npts)
    model.add_verax_restraints(verax_restraints)
    model.add_lator_restraints(lator_restraints)
    model.add_nodal_loads(nodal_loads)

    StaticSolver(model).solve()

    stab = StabilitySolver(model)
    stab.solve()
    return stab.mu_crs[0]


def rel_error(mu, ref):
    return np.abs(np.asarray(mu) - ref) / ref * 100


# ── 1. Cuadratura: 4 elementos, reglas de 2 a 8 puntos ────────────────────────
n_fixed = 4
gauss_points = [2, 3, 4, 5, 6, 7, 8]

ref_gauss = {c: run_model(c, n_fixed, npts=12) for c in sections}
mu_gauss = {c: [run_model(c, n_fixed, npts=g) for g in gauss_points] for c in sections}
err_gauss = {c: rel_error(mu_gauss[c], ref_gauss[c]) for c in sections}

print(f"1. Quadrature: {n_fixed} elements, reference = 12-point rule")
print(f"   μ_ref(1a) = {ref_gauss['1a']:.8f}   μ_ref(1b) = {ref_gauss['1b']:.8f}\n")
print(f"{'points':>6}  {'μ_1a':>12}  {'err_1a %':>10}  {'μ_1b':>12}  {'err_1b %':>10}")
print("-" * 60)
for i, g in enumerate(gauss_points):
    print(f"{g:>6}  {mu_gauss['1a'][i]:>12.6f}  {err_gauss['1a'][i]:>10.2e}"
          f"  {mu_gauss['1b'][i]:>12.6f}  {err_gauss['1b'][i]:>10.2e}")


# ── 2. Discretización: regla de 4 puntos, malla creciente ─────────────────────
mesh_sizes = [2, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100]

ref_mesh = {c: run_model(c, 200) for c in sections}
mu_mesh = {c: [run_model(c, n) for n in mesh_sizes] for c in sections}
err_mesh = {c: rel_error(mu_mesh[c], ref_mesh[c]) for c in sections}

print(f"\n2. Mesh: 4-point rule, reference = 200 elements")
print(f"   μ_ref(1a) = {ref_mesh['1a']:.8f}   μ_ref(1b) = {ref_mesh['1b']:.8f}\n")
print(f"{'n':>6}  {'μ_1a':>12}  {'err_1a %':>10}  {'μ_1b':>12}  {'err_1b %':>10}")
print("-" * 60)
for i, n in enumerate(mesh_sizes):
    print(f"{n:>6}  {mu_mesh['1a'][i]:>12.6f}  {err_mesh['1a'][i]:>10.2e}"
          f"  {mu_mesh['1b'][i]:>12.6f}  {err_mesh['1b'][i]:>10.2e}")

ng = np.array(gauss_points)
ns = np.array(mesh_sizes)

# ── Plots ─────────────────────────────────────────────────────────────────────
plt.style.use(["science", "notebook", "grid",
               {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True}])
fs_axes, fs_ticks, fs_legend = 16, 14, 12
n_ticks = [2, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Error de cuadratura vs puntos de Gauss
ax = axes[0]
ax.plot(ng, err_gauss["1a"], "o-", color="blue", lw=1.5, ms=6, label="Doubly symmetric (1a)")
ax.plot(ng, err_gauss["1b"], "s--", color="red", lw=1.5, ms=8, mfc="none", mew=1.2, label="Singly symmetric (1b)")
ax.set_xlabel("Number of Gauss points", fontsize=fs_axes)
ax.set_xticks(gauss_points)

# Error de discretización vs n
ax = axes[1]
ax.plot(ns, err_mesh["1a"], "o-", color="blue", lw=1.5, ms=6, label="Doubly symmetric (1a)")
ax.plot(ns, err_mesh["1b"], "s--", color="red", lw=1.5, ms=8, mfc="none", mew=1.2, label="Singly symmetric (1b)")
ax.set_xlabel("Number of elements $n$", fontsize=fs_axes)
ax.set_xscale("log")
ax.set_xticks(n_ticks, labels=[str(n) for n in n_ticks])
ax.tick_params(axis="x", which="minor", labelbottom=False)

for ax in axes:
    ax.set_ylabel("Error [%]", fontsize=fs_axes)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=fs_ticks)
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95), fancybox=False, edgecolor="black",
              fontsize=fs_legend)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig(r"D:\Maestria UFRGS\Tesis maestria\Disertacion\fig\convergence_plot3.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\nDone.")
