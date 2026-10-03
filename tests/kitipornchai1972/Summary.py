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
from matplotlib.lines import Line2D
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── helpers ────────────────────────────────────────────────────────────────────

def section(hc, B, T):
    return ISection_MS(h=hc + T, bf1=B, bf2=B, tw=t, tf1=T, tf2=T, r1=0, r2=0, It_type="plates")


def make_mesh_full(sec_min, sec_max, L, nelems):
    """Double-taper: min → max → min over the full span."""
    nnods  = nelems + 1                      # nelems par: el nodo nelems//2 está en el centro
    coords = np.linspace(0, L, nnods)
    norm   = coords / L
    half   = nnods // 2
    secs   = (interpolate_multiple_sections(sec_min, sec_max, norm[:half+1] * 2.0) +
              interpolate_multiple_sections(sec_max, sec_min, (norm[half+1:] - 0.5) * 2.0))
    edata  = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, secs, edata


def solve(coords, sections, edata, nodal_loads):
    nelems = len(edata)
    model = StabilityModel()
    model.add_materials([Material(E=65160e6, nu=65160/(2*25650) - 1, rho=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata)
    model.add_verax_restraints(np.array([[0, 1, 1, 0], [nelems, 0, 1, 0]]))
    model.add_lator_restraints(np.array([[0, 1, 0, 1, 0], [nelems, 1, 0, 1, 0]]))
    model.add_nodal_loads(nodal_loads)

    s1 = StaticSolver(model);    s1.solve()
    s2 = StabilitySolver(model); s2.solve()
    return s1.max_vals(), s2.mu_crs[0]


def delta(ref, val):
    return (ref - val) / val * 100


def print_header(title, param_label):
    print("\n" + "═" * 92)
    print(f"  {title}")
    print("═" * 92)
    print(f"  {param_label:>8}  {'Reference':>12}  {'LTBeamN':>15}  {'Δ %':>8}"
          f"  {'PyLTB':>14}  {'Δ %':>8}  {'ΔLTB %':>11}")
    print("  " + "─" * 88)


def print_row(label, mu, ref, ltb):
    if np.isnan(ltb):   # LTBeamN no resuelve
        print(f"  {label:>8}  {ref:>12.1f}  {'—':>15}  {'—':>8}"
              f"  {mu:>14.1f}  {delta(ref, mu):>7.2f}%  {'—':>11}")
        return
    print(f"  {label:>8}  {ref:>12.1f}  {ltb:>15.1f}  {delta(ref, ltb):>7.2f}%"
          f"  {mu:>14.1f}  {delta(ref, mu):>7.2f}%  {delta(ltb, mu):>10.2f}%")


# ── data ───────────────────────────────────────────────────────────────────────

L  = 1.524                                    # [m]
hc, B, T, t = 0.0728, 0.0316, 0.00311, 0.00213  # [m], hc entre centroides de alas
nelems = 20

alphas  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sec_max = section(hc, B, T)

# Carga central de 1 N en el centroide del ala superior: W_cr = mu_cr [N]
loads = np.array([[nelems//2, 0, 3, 0.0, -T/2, 0.0, -1.0, 0.0]])

# Ref.: puntos de Bradford y Cuk (1988), Fig. 5, leidos del grafico (±1 %)
# LTBeamN 2.0.2: media viga con simetria, It de Villette
cases = {
    "depth": {
        "title":   "Depth tapered      –  hc → alpha_h hc at the supports",
        "label":   "alpha_h",
        "sec_min": lambda a: section(a*hc, B, T),
        "refs":    [892, 870, 864, 864, 873, 887, 907, 930, 956],
        "ltbeamn": [879.8, 853.2, 841.9, 841.5, 849.4, 863.6, 883.3, 907.2, 935.0],
        # Trahair (2014), Fig. 7, leido del grafico: curva FTBTM y ensayos, en Q/Q_u
        "ftbtm":   ([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.901, 0.894, 0.884, 0.879, 0.887, 0.898, 0.917, 0.939, 0.965]),
        "tests":   ([0.125, 0.125, 0.27, 0.27, 0.55, 0.55],
                    [0.930, 0.900, 0.897, 0.866, 0.927, 0.895]),
        "style":   ("#2a78d6", "-"),
    },
    "width": {
        "title":   "Width tapered      –  B → alpha_B B at the supports",
        "label":   "alpha_B",
        "sec_min": lambda a: section(hc, a*B, T),
        "refs":    [442, 519, 587, 649, 711, 766, 823, 877, 933],
        "ltbeamn": [429.6, 501.9, 568.5, 631.1, 691.0, 748.8, 805.0, 859.8, 913.5],
        "ftbtm":   ([0.15, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9],
                    [0.490, 0.524, 0.660, 0.717, 0.781, 0.832, 0.913, 0.942]),
        "tests":   ([0.30, 0.30, 0.60, 0.80, 0.80],
                    [0.569, 0.549, 0.760, 0.897, 0.867]),
        "style":   ("#eb6834", ":"),
    },
    "thickness": {
        "title":   "Thickness tapered  –  T → alpha_T T at the supports",
        "label":   "alpha_T",
        "sec_min": lambda a: section(hc, B, a*T),
        "refs":    [520, 543, 574, 607, 652, 706, 769, 832, 906],
        "ltbeamn": [np.nan, 229.3, 395.5, 492.1, 572.7, 649.0, 725.2, 826.3, 893.6],
        "ftbtm":   ([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.553, 0.621, 0.669, 0.723, 0.782, 0.842, 0.919]),
        "tests":   ([0.16, 0.16, 0.48, 0.48, 0.81, 0.81],
                    [0.538, 0.512, 0.645, 0.623, 0.869, 0.845]),
        "style":   ("#1baf7a", "--"),
    },
}
ltbeamn_u = 966.3   # LTBeamN, viga uniforme (alpha = 1)


# ── Kitipornchai and Trahair (1972) ────────────────────────────────────────────

for case in cases.values():
    print_header(case["title"], case["label"])
    for a, ref, ltb in zip(alphas, case["refs"], case["ltbeamn"]):
        coords, sections, edata = make_mesh_full(case["sec_min"](a), sec_max, L, nelems)
        _, mu = solve(coords, sections, edata, loads)
        print_row(f"{a:.1f}", mu, ref, ltb)

print("\n  W_cr [N]. Ref.: Bradford y Cuk (1988), Fig. 5, leido del grafico (±1 %)")
print("  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100")
print("  LTBeamN usa It de Villette; con espesor variable y alpha_T <= 0.7 toma I_wpsi = dIw/dx")
print("═" * 92 + "\n")


# ── Plots: Bradford y Cuk (1988), Fig. 5, y Trahair (2014), Fig. 7 ───────────────

alphas_fine = np.round(np.arange(0.1, 1.0001, 0.05), 2)
coords, sections, edata = make_mesh_full(sec_max, sec_max, L, nelems)
_, mu_u = solve(coords, sections, edata, loads)        # viga uniforme (alpha = 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for name, case in cases.items():
    color, ls = case["style"]
    mus = []
    for a in alphas_fine:
        coords, sections, edata = make_mesh_full(case["sec_min"](a), sec_max, L, nelems)
        mus.append(solve(coords, sections, edata, loads)[1])
    mus = np.array(mus)
    ltb = np.array(case["ltbeamn"])
    ax1.plot(alphas_fine, mus, color=color, ls=ls, lw=1.5)
    ax1.plot(alphas, case["refs"], "o", color=color, ms=6)
    ax1.plot(alphas, ltb, "x", color=color, ms=7, mew=1.5)
    ax2.plot(alphas_fine, mus / mu_u, color=color, ls=ls, lw=1.5)
    ax2.plot(*case["ftbtm"], "D", color=color, ms=6, mfc="none", mew=1.2)
    ax2.plot(*case["tests"], "s", color=color, ms=7, mfc="none", mew=1.2)
    ax2.plot(alphas, ltb / ltbeamn_u, "x", color=color, ms=7, mew=1.5)

cases_leg = [Line2D([], [], color=c["style"][0], ls=c["style"][1], lw=1.5, label=f"PyLTB, {n}")
             for n, c in cases.items()]
gray = "#52514e"
src_1 = [Line2D([], [], ls="", marker="o", color=gray, label="Bradford and Cuk (1988)"),
         Line2D([], [], ls="", marker="x", color=gray, mew=1.5, label="LTBeamN")]
src_2 = [Line2D([], [], ls="", marker="D", color=gray, mfc="none", label="Trahair (2014), FTBTM"),
         Line2D([], [], ls="", marker="s", color=gray, mfc="none", label="Test (Trahair 2014)"),
         Line2D([], [], ls="", marker="x", color=gray, mew=1.5, label="LTBeamN")]
for ax, src, ylabel, ytop in ((ax1, src_1, r"$W_{cr}$ [N]", 1000), (ax2, src_2, r"$Q/Q_u$", 1.0)):
    ax.set_xlim(0, 1); ax.set_ylim(0, ytop * 1.05)
    ax.set_xlabel(r"Taper constant $\alpha_h$, $\alpha_B$, $\alpha_T$"); ax.set_ylabel(ylabel)
    ax.grid(color="#d9d9d6", lw=0.6)
    ax.legend(handles=cases_leg + src, loc="lower right", fontsize=8, frameon=False)
ax1.set_title("Critical load (Bradford and Cuk 1988, Fig. 5)", fontsize=10)
ax2.set_title("Dimensionless load (Trahair 2014, Fig. 7)", fontsize=10)
fig.tight_layout()
plt.show()
