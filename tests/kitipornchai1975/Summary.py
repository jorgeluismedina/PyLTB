"""
Kitipornchai and Trahair (1975)  –  Simply supported monosymmetric width-tapered beams
                                    Mid-span point load at top flange centroid
  (a):  top flange B → alpha_B B, bottom flange alpha_B B → B  (mid-span → supports)
  (b):  the same beam upside down

Geometry from Bradford and Cuk (1988), Fig. 6: the maximum dimensions and material of their
Fig. 5, L = 1524 mm, hc = 72.8 (between flange centroids, constant), B = 31.6, T = 3.11,
t = 2.13 mm; E = 65160 MPa, G = 25650 MPa.

Δ = (Ref - value) / value * 100, the definition of Beyer (2015).
"""


import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── Datos ──────────────────────────────────────────────────────────────────────

L  = 1.524                                        # [m]
hc, B, T, t = 0.0728, 0.0316, 0.00311, 0.00213    # [m], hc entre centroides de alas
nelems   = 20                                     # par: el nodo nelems//2 está en el centro
material = Material(E=65160e6, nu=65160/(2*25650) - 1, rho=1.0)   # G = 25650 MPa

# Carga central de 1 N en el centroide del ala superior: W_cr = mu_cr [N]
loads = np.array([[nelems//2, 0, 3, 0.0, -T/2, 0.0, -1.0, 0.0]])

alphas = np.linspace(0.1, 1.0, 19)    # paso 0.05; alpha = 1 es la viga uniforme
tab    = slice(0, -1, 2)              # alphas de la tabla: 0.1, 0.2, ..., 0.9

# Ref.: puntos de Bradford y Cuk (1988), Fig. 6, leidos del grafico (±1 %)
# LTBeamN 2.0.2: media viga con simetria, It de Villette
cases = {
    "a": {
        "title":   "(a) top flange wide at mid-span   –  B → alpha_B B at the supports",
        "secs":    lambda a: (section(a*B, B), section(B, a*B)),       # (apoyo, centro)
        "refs":    np.array([507, 555, 613, 658, 722, 771, 821, 876, 926]),   # 0.1 y 0.2 bajo los cuadrados de H&T
        "ltbeamn": np.array([122.2, 148.1, 181.0, 224.4, 282.5, 359.8, 461.4, 592.5, 758.6]),
        # Hancock y Trahair (1978), elementos uniformes escalonados, en la Fig. 6 de Bradford y Cuk
        "ht":      np.array([510, 558, 620, 671, 734, 785, 841, 898, 939]),
        # Kitipornchai y Trahair (1975): curva de la Fig. 6 de Bradford y Cuk [N]
        "kt":      ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [526, 602, 673, 737, 796, 848, 892, 928, 954]),
        # Trahair (2014), Fig. 8, en Q/Q_u: analisis de Kitipornchai y Trahair (1975), FTBTM y ensayo
        "kt_q":    ([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                    [0.643, 0.708, 0.768, 0.817, 0.864, 0.905, 0.945, 0.982]),
        "ftbtm":   ([0.2, 0.4, 0.6, 0.8], [0.630, 0.752, 0.847, 0.925]),
        "tests":   ([0.57], [0.829]),
        "color":   "red",
    },
    "b": {
        "title":   "(b) top flange narrow at mid-span –  alpha_B B → B at the supports",
        "secs":    lambda a: (section(B, a*B), section(a*B, B)),
        "refs":    np.array([279, 307, 346, 397, 466, 547, 636, 739, 852]),
        "ltbeamn": np.array([694.0, 868.3, 1045.9, 1151.8, 1180.4, 1161.6, 1119.3, 1067.2, 1014.5]),
        "ht":      None,
        "kt":      ([0.15, 0.25, 0.35, 0.45, 0.55, 0.75, 0.85, 0.95],
                    [292, 324, 368, 431, 502, 686, 791, 909]),
        "kt_q":    ([0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.85, 0.95],
                    [0.319, 0.364, 0.415, 0.484, 0.567, 0.622, 0.791, 0.916]),
        "ftbtm":   ([0.2, 0.4, 0.6, 0.8], [0.284, 0.373, 0.514, 0.726]),
        "tests":   ([0.57], [0.500]),
        "color":   "blue",
    },
}
ltbeamn_u = 965.8   # LTBeamN, viga uniforme (alpha = 1)


# ── Modelo ─────────────────────────────────────────────────────────────────────

def section(b_top, b_bot):
    return ISection_MS(h=hc + T, bf1=b_top, bf2=b_bot, tw=t, tf1=T, tf2=T, r1=0, r2=0, It_type="plates")


def solve(sec_end, sec_mid):
    """mu_cr de la viga con doble ahusamiento: sec_end en los apoyos y sec_mid en el centro."""
    coords = np.linspace(0, L, nelems + 1)
    left   = interpolate_multiple_sections(sec_end, sec_mid, 2 * coords[:nelems//2 + 1] / L)
    edata  = np.array([[1, 0, e, e + 1] for e in range(nelems)])

    model = StabilityModel()
    model.add_materials([material])
    model.add_sections(left + left[-2::-1])          # mitad derecha simétrica
    model.add_nodes(coords)
    model.add_tapered_elements(edata, align=3)       # fibra superior recta: canto constante
    model.add_verax_restraints(np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]]))
    model.add_lator_restraints(np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]]))
    model.add_nodal_loads(loads)

    StaticSolver(model).solve()
    stab = StabilitySolver(model)
    stab.solve()
    return stab.mu_crs[0]


# ── Resolución ─────────────────────────────────────────────────────────────────

for case in cases.values():
    case["mu"] = np.array([solve(*case["secs"](a)) for a in alphas])
mu_u = cases["a"]["mu"][-1]     # alpha = 1: la viga uniforme, igual en los dos casos


# ── Tablas ─────────────────────────────────────────────────────────────────────

def delta(ref, val):
    return (ref - val) / val * 100


for case in cases.values():
    ref, ltb, mu = case["refs"], case["ltbeamn"], case["mu"][tab]
    print("\n" + "═" * 80)
    print(f"  {case['title']}")
    print("═" * 80)
    print(f"  {'alpha':>6}  {'Ref.':>10}  {'LTBeamN':>10}  {'Δ %':>8}  {'PyLTB':>10}  {'Δ %':>8}"
          f"  {'ΔLTB %':>8}")
    print("  " + "─" * 76)
    for a, r, l, m in zip(alphas[tab], ref, ltb, mu):
        print(f"  {a:6.1f}  {r:10.1f}  {l:10.1f}  {delta(r, l):8.2f}"
              f"  {m:10.1f}  {delta(r, m):8.2f}  {delta(l, m):8.2f}")

print("\n  W_cr [N]. Ref.: Bradford y Cuk (1988), Fig. 6, leido del grafico (±1 %)")
print("  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100")
print("  En (a) Bradford y Cuk y Hancock y Trahair (1978) quedan por debajo de Kitipornchai y")
print("  Trahair (1975) y de Trahair (2014); en (b) coinciden con Kitipornchai y Trahair (1975).")
print("  LTBeamN usa It de Villette y anula I_psi e I_ypsi; en (b) supera a la viga uniforme")
print("═" * 80 + "\n")


# ── Gráficos: Bradford y Cuk (1988), Fig. 6, y Trahair (2014), Fig. 8 ──────────

plt.style.use(["science", "notebook", "grid",
               {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True}])
fs_axes, fs_ticks, fs_legend, fs_title = 16, 14, 12, 15
gray = "#52514e"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Cada fuente se dibuja con el color del caso; en la leyenda va una sola entrada gris por
# fuente, que se agrega con un plot sin datos: ax.plot([], [], ...). Los dos graficos
# comparten una leyenda bajo la figura: cada entrada se define una sola vez

# W_cr [N]: Bradford y Cuk (1988), Fig. 6
for name, case in cases.items():
    c = case["color"]
    ax1.plot(alphas, case["mu"], "-", color=c, lw=1.5, label=f"PyLTB, ({name})")
    ax1.plot(alphas[tab], case["refs"], "o", color=c, ms=6)
    if case["ht"] is not None:
        ax1.plot(alphas[tab], case["ht"], "s", color=c, ms=6)
    ax1.plot(*case["kt"], "+", color=c, ms=9, mew=1.5)
    ax1.plot(alphas[tab], case["ltbeamn"], "x", color=c, ms=7, mew=1.5)
ax1.plot([], [], "o", color=gray, ms=6, label="Bradford and Cuk (1988)")
ax1.plot([], [], "s", color=gray, ms=6, label="Hancock and Trahair (1978)")
ax1.plot([], [], "+", color=gray, ms=9, mew=1.5, label="Kitipornchai and Trahair (1975)")
ax1.plot([], [], "x", color=gray, ms=7, mew=1.5, label="LTBeamN")
ax1.set_ylim(0, 1250)
ax1.set_title("Critical load (Bradford and Cuk 1988, Fig. 6)", fontsize=fs_title)
ax1.set_ylabel(r"$W_{cr}$ [N]", fontsize=fs_axes)

# Q/Q_u: Trahair (2014), Fig. 8
for name, case in cases.items():
    c = case["color"]
    ax2.plot(alphas, case["mu"] / mu_u, "-", color=c, lw=1.5)
    ax2.plot(*case["kt_q"], "+", color=c, ms=9, mew=1.5)
    ax2.plot(*case["ftbtm"], "D", color=c, ms=6, mfc="none", mew=1.2)
    ax2.plot(*case["tests"], "s", color=c, ms=7, mfc="none", mew=1.2)
    ax2.plot(alphas[tab], case["ltbeamn"] / ltbeamn_u, "x", color=c, ms=7, mew=1.5)
ax2.plot([], [], "D", color=gray, ms=6, mfc="none", mew=1.2, label="Trahair (2014), FTBTM")
ax2.plot([], [], "s", color=gray, ms=7, mfc="none", mew=1.2, label="Test (Trahair 2014)")
ax2.set_ylim(0, 1.3)
ax2.set_title("Dimensionless load (Trahair 2014, Fig. 8)", fontsize=fs_title)
ax2.set_ylabel(r"$Q/Q_u$", fontsize=fs_axes)

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Flange taper constant $\alpha_B$", fontsize=fs_axes)
    ax.tick_params(axis="both", which="major", labelsize=fs_ticks)
    ax.grid(True, alpha=0.3)

handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
fig.legend(handles=handles, loc="lower center", ncol=4, fancybox=False, edgecolor="black",
           fontsize=fs_legend)
plt.tight_layout(rect=(0, 0.13, 1, 1))     # deja lugar a la leyenda
plt.show()
