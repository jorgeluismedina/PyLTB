"""
Kitipornchai and Trahair (1972)  –  Simply supported bisymmetric tapered beams
                                    Mid-span point load at top flange centroid
  depth:      flange centroid distance hc  → alpha_h * hc at the supports
  width:      flange width B               → alpha_B * B  at the supports
  thickness:  flange thickness T           → alpha_T * T  at the supports

Geometry from Bradford and Cuk (1988), Fig. 5: L = 1524 mm; at mid-span hc = 72.8 (between
flange centroids), B = 31.6, T = 3.11, t = 2.13 mm; E = 65160 MPa, G = 25650 MPa.

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

# Ref.: puntos de Bradford y Cuk (1988), Fig. 5, leidos del grafico (±1 %)
# LTBeamN 2.0.2: media viga con simetria, It de Villette
# Trahair (2014), Fig. 7, leido del grafico: curva FTBTM y ensayos, en Q/Q_u
cases = {
    "depth": {
        "title":   "Depth tapered      –  hc → alpha_h hc at the supports",
        "sec_end": lambda a: section(a*hc, B, T),
        "refs":    np.array([892, 870, 864, 864, 873, 887, 907, 930, 956]),
        "ltbeamn": np.array([879.8, 853.2, 841.9, 841.5, 849.4, 863.6, 883.3, 907.2, 935.0]),
        "ftbtm":   ([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.901, 0.894, 0.884, 0.879, 0.887, 0.898, 0.917, 0.939, 0.965]),
        "tests":   ([0.125, 0.125, 0.27, 0.27, 0.55, 0.55],
                    [0.930, 0.900, 0.897, 0.866, 0.927, 0.895]),
        "color":   "red",
    },
    "width": {
        "title":   "Width tapered      –  B → alpha_B B at the supports",
        "sec_end": lambda a: section(hc, a*B, T),
        "refs":    np.array([442, 519, 587, 649, 711, 766, 823, 877, 933]),
        "ltbeamn": np.array([429.6, 501.9, 568.5, 631.1, 691.0, 748.8, 805.0, 859.8, 913.5]),
        "ftbtm":   ([0.15, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9],
                    [0.490, 0.524, 0.660, 0.717, 0.781, 0.832, 0.913, 0.942]),
        "tests":   ([0.30, 0.30, 0.60, 0.80, 0.80],
                    [0.569, 0.549, 0.760, 0.897, 0.867]),
        "color":   "blue",
    },
    "thickness": {
        "title":   "Thickness tapered  –  T → alpha_T T at the supports",
        "sec_end": lambda a: section(hc, B, a*T),
        "refs":    np.array([520, 543, 574, 607, 652, 706, 769, 832, 906]),
        "ltbeamn": np.array([np.nan, 229.3, 395.5, 492.1, 572.7, 649.0, 725.2, 826.3, 893.6]),
        "ftbtm":   ([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.553, 0.621, 0.669, 0.723, 0.782, 0.842, 0.919]),
        "tests":   ([0.16, 0.16, 0.48, 0.48, 0.81, 0.81],
                    [0.538, 0.512, 0.645, 0.623, 0.869, 0.845]),
        "color":   "darkgreen",
    },
}
ltbeamn_u = 966.3   # LTBeamN, viga uniforme (alpha = 1)


# ── Modelo ─────────────────────────────────────────────────────────────────────

def section(hc, B, T):
    return ISection_MS(h=hc + T, bf1=B, bf2=B, tw=t, tf1=T, tf2=T, r1=0, r2=0, It_type="plates")


def solve(sec_end, sec_mid):
    """mu_cr de la viga con doble ahusamiento: sec_end en los apoyos y sec_mid en el centro."""
    coords = np.linspace(0, L, nelems + 1)
    left   = interpolate_multiple_sections(sec_end, sec_mid, 2 * coords[:nelems//2 + 1] / L)
    edata  = np.array([[1, 0, e, e + 1] for e in range(nelems)])

    model = StabilityModel()
    model.add_materials([material])
    model.add_sections(left + left[-2::-1])          # mitad derecha simétrica
    model.add_nodes(coords)
    model.add_tapered_elements(edata)
    model.add_verax_restraints(np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]]))
    model.add_lator_restraints(np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]]))
    model.add_nodal_loads(loads)

    StaticSolver(model).solve()
    stab = StabilitySolver(model)
    stab.solve()
    return stab.mu_crs[0]


# ── Resolución ─────────────────────────────────────────────────────────────────

sec_mid = section(hc, B, T)
for case in cases.values():
    case["mu"] = np.array([solve(case["sec_end"](a), sec_mid) for a in alphas])
mu_u = cases["depth"]["mu"][-1]     # alpha = 1: la viga uniforme, igual en los tres casos


# ── Tablas ─────────────────────────────────────────────────────────────────────

def delta(ref, val):
    return (ref - val) / val * 100


def num(x, spec):
    """x con formato spec; '—' donde LTBeamN no resuelve (nan)."""
    s = f"{x:{spec}}"
    return s if np.isfinite(x) else "—".rjust(len(s))


for case in cases.values():
    ref, ltb, mu = case["refs"], case["ltbeamn"], case["mu"][tab]
    print("\n" + "═" * 80)
    print(f"  {case['title']}")
    print("═" * 80)
    print(f"  {'alpha':>6}  {'Ref.':>10}  {'LTBeamN':>10}  {'Δ %':>8}  {'PyLTB':>10}  {'Δ %':>8}"
          f"  {'ΔLTB %':>8}")
    print("  " + "─" * 76)
    for a, r, l, m in zip(alphas[tab], ref, ltb, mu):
        print(f"  {a:6.1f}  {r:10.1f}  {num(l, '10.1f')}  {num(delta(r, l), '8.2f')}"
              f"  {m:10.1f}  {delta(r, m):8.2f}  {num(delta(l, m), '8.2f')}")

print("\n  W_cr [N]. Ref.: Bradford y Cuk (1988), Fig. 5, leido del grafico (±1 %)")
print("  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100")
print("  LTBeamN usa It de Villette; con espesor variable y alpha_T <= 0.7 toma I_wpsi = dIw/dx")
print("═" * 80 + "\n")


# ── Gráficos: Bradford y Cuk (1988), Fig. 5, y Trahair (2014), Fig. 7 ──────────

plt.style.use(["science", "notebook", "grid",
               {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True}])
fs_axes, fs_ticks, fs_legend, fs_title = 16, 14, 12, 15
gray = "#52514e"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Cada fuente se dibuja con el color del caso; en la leyenda va una sola entrada gris por
# fuente, que se agrega con un plot sin datos: ax.plot([], [], ...)

# W_cr [N]: Bradford y Cuk (1988), Fig. 5
for name, case in cases.items():
    c = case["color"]
    ax1.plot(alphas, case["mu"], "-", color=c, lw=1.5, label=f"PyLTB, {name}")
    ax1.plot(alphas[tab], case["refs"], "o", color=c, ms=6)
    ax1.plot(alphas[tab], case["ltbeamn"], "x", color=c, ms=7, mew=1.5)
ax1.plot([], [], "o", color=gray, ms=6, label="Bradford and Cuk (1988)")
ax1.plot([], [], "x", color=gray, ms=7, mew=1.5, label="LTBeamN")
ax1.set_ylim(0, 1050)
ax1.set_title("Critical load (Bradford and Cuk 1988, Fig. 5)", fontsize=fs_title)
ax1.set_ylabel(r"$W_{cr}$ [N]", fontsize=fs_axes)

# Q/Q_u: Trahair (2014), Fig. 7
for name, case in cases.items():
    c = case["color"]
    ax2.plot(alphas, case["mu"] / mu_u, "-", color=c, lw=1.5, label=f"PyLTB, {name}")
    ax2.plot(*case["ftbtm"], "D", color=c, ms=6, mfc="none", mew=1.2)
    ax2.plot(*case["tests"], "s", color=c, ms=7, mfc="none", mew=1.2)
    ax2.plot(alphas[tab], case["ltbeamn"] / ltbeamn_u, "x", color=c, ms=7, mew=1.5)
ax2.plot([], [], "D", color=gray, ms=6, mfc="none", mew=1.2, label="Trahair (2014), FTBTM")
ax2.plot([], [], "s", color=gray, ms=7, mfc="none", mew=1.2, label="Test (Trahair 2014)")
ax2.plot([], [], "x", color=gray, ms=7, mew=1.5, label="LTBeamN")
ax2.set_ylim(0, 1.05)
ax2.set_title("Dimensionless load (Trahair 2014, Fig. 7)", fontsize=fs_title)
ax2.set_ylabel(r"$Q/Q_u$", fontsize=fs_axes)

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Taper constant $\alpha_h$, $\alpha_B$, $\alpha_T$", fontsize=fs_axes)
    ax.tick_params(axis="both", which="major", labelsize=fs_ticks)
    ax.legend(loc="lower right", fancybox=False, edgecolor="black", fontsize=fs_legend)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
