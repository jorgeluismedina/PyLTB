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
from matplotlib.lines import Line2D
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── helpers ────────────────────────────────────────────────────────────────────

def section(b_top, b_bot):
    return ISection_MS(h=hc + T, bf1=b_top, bf2=b_bot, tw=t, tf1=T, tf2=T, r1=0, r2=0, It_type="plates")


def make_mesh_full(sec_end, sec_mid, L, nelems):
    """Double-taper: end → mid-span → end over the full span."""
    nnods  = nelems + 1                      # nelems par: el nodo nelems//2 está en el centro
    coords = np.linspace(0, L, nnods)
    norm   = coords / L
    half   = nnods // 2
    secs   = (interpolate_multiple_sections(sec_end, sec_mid, norm[:half+1] * 2.0) +
              interpolate_multiple_sections(sec_mid, sec_end, (norm[half+1:] - 0.5) * 2.0))
    edata  = np.array([[1, 0, e, e+1] for e in range(nelems)])
    return coords, secs, edata


def solve(coords, sections, edata, nodal_loads):
    nelems = len(edata)
    model = StabilityModel()
    model.add_materials([Material(E=65160e6, nu=65160/(2*25650) - 1, rho=1.0)])
    model.add_sections(sections)
    model.add_nodes(coords)
    model.add_tapered_elements(edata, align=3)   # fibra superior recta: canto constante
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
    print(f"  {label:>8}  {ref:>12.1f}  {ltb:>15.1f}  {delta(ref, ltb):>7.2f}%"
          f"  {mu:>14.1f}  {delta(ref, mu):>7.2f}%  {delta(ltb, mu):>10.2f}%")


# ── data ───────────────────────────────────────────────────────────────────────

L  = 1.524                                    # [m]
hc, B, T, t = 0.0728, 0.0316, 0.00311, 0.00213  # [m], hc entre centroides de alas
nelems = 20

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Carga central de 1 N en el centroide del ala superior: W_cr = mu_cr [N]
loads = np.array([[nelems//2, 0, 3, 0.0, -T/2, 0.0, -1.0, 0.0]])

# Ref.: puntos de Bradford y Cuk (1988), Fig. 6, leidos del grafico (±1 %)
# LTBeamN 2.0.2: media viga con simetria, It de Villette
cases = {
    "a": {
        "title":   "(a) top flange wide at mid-span   –  B → alpha_B B at the supports",
        "secs":    lambda a: (section(a*B, B), section(B, a*B)),        # (apoyo, centro)
        "refs":    [507, 555, 613, 658, 722, 771, 821, 876, 926],       # 0.1 y 0.2 bajo los cuadrados de H&T
        "ltbeamn": [122.2, 148.1, 181.0, 224.4, 282.5, 359.8, 461.4, 592.5, 758.6],
        # Hancock y Trahair (1978), elementos uniformes escalonados, en la Fig. 6 de Bradford y Cuk
        "ht":      [510, 558, 620, 671, 734, 785, 841, 898, 939],
        # Kitipornchai y Trahair (1975): curva de la Fig. 6 de Bradford y Cuk [N]
        "kt":      ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [526, 602, 673, 737, 796, 848, 892, 928, 954]),
        # Trahair (2014), Fig. 8, en Q/Q_u: analisis de Kitipornchai y Trahair (1975), FTBTM y ensayo
        "kt_q":    ([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                    [0.643, 0.708, 0.768, 0.817, 0.864, 0.905, 0.945, 0.982]),
        "ftbtm":   ([0.2, 0.4, 0.6, 0.8], [0.630, 0.752, 0.847, 0.925]),
        "tests":   ([0.57], [0.829]),
        "style":   ("#2a78d6", "-"),
    },
    "b": {
        "title":   "(b) top flange narrow at mid-span –  alpha_B B → B at the supports",
        "secs":    lambda a: (section(B, a*B), section(a*B, B)),
        "refs":    [279, 307, 346, 397, 466, 547, 636, 739, 852],
        "ltbeamn": [694.0, 868.3, 1045.9, 1151.8, 1180.4, 1161.6, 1119.3, 1067.2, 1014.5],
        "ht":      None,
        "kt":      ([0.15, 0.25, 0.35, 0.45, 0.55, 0.75, 0.85, 0.95],
                    [292, 324, 368, 431, 502, 686, 791, 909]),
        "kt_q":    ([0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.85, 0.95],
                    [0.319, 0.364, 0.415, 0.484, 0.567, 0.622, 0.791, 0.916]),
        "ftbtm":   ([0.2, 0.4, 0.6, 0.8], [0.284, 0.373, 0.514, 0.726]),
        "tests":   ([0.57], [0.500]),
        "style":   ("#eb6834", ":"),
    },
}
ltbeamn_u = 965.8   # LTBeamN, viga uniforme (alpha = 1)


# ── Kitipornchai and Trahair (1975) ────────────────────────────────────────────

for case in cases.values():
    print_header(case["title"], "alpha_B")
    for a, ref, ltb in zip(alphas, case["refs"], case["ltbeamn"]):
        coords, sections, edata = make_mesh_full(*case["secs"](a), L, nelems)
        _, mu = solve(coords, sections, edata, loads)
        print_row(f"{a:.1f}", mu, ref, ltb)

print("\n  W_cr [N]. Ref.: Bradford y Cuk (1988), Fig. 6, leido del grafico (±1 %)")
print("  Δ %    = (Ref. - valor) / valor * 100, como en Beyer (2015)")
print("  ΔLTB % = (LTBeamN - PyLTB) / PyLTB * 100")
print("  En (a) Bradford y Cuk y Hancock y Trahair (1978) quedan por debajo de Kitipornchai y")
print("  Trahair (1975) y de Trahair (2014); en (b) coinciden con Kitipornchai y Trahair (1975).")
print("  LTBeamN usa It de Villette y anula I_psi e I_ypsi; en (b) supera a la viga uniforme")
print("═" * 92 + "\n")


# ── Plots: Bradford y Cuk (1988), Fig. 6, y Trahair (2014), Fig. 8 ───────────────

alphas_fine = np.round(np.arange(0.1, 1.0001, 0.05), 2)
sec_u = section(B, B)
coords, sections, edata = make_mesh_full(sec_u, sec_u, L, nelems)
_, mu_u = solve(coords, sections, edata, loads)        # viga uniforme (alpha = 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for name, case in cases.items():
    color, ls = case["style"]
    mus = []
    for a in alphas_fine:
        coords, sections, edata = make_mesh_full(*case["secs"](a), L, nelems)
        mus.append(solve(coords, sections, edata, loads)[1])
    mus = np.array(mus)
    ltb = np.array(case["ltbeamn"])
    ax1.plot(alphas_fine, mus, color=color, ls=ls, lw=1.5)
    ax1.plot(alphas, case["refs"], "o", color=color, ms=6)
    if case["ht"]:
        ax1.plot(alphas, case["ht"], "s", color=color, ms=6)
    ax1.plot(*case["kt"], "+", color=color, ms=9, mew=1.5)
    ax1.plot(alphas, ltb, "x", color=color, ms=7, mew=1.5)
    ax2.plot(alphas_fine, mus / mu_u, color=color, ls=ls, lw=1.5)
    ax2.plot(*case["kt_q"], "+", color=color, ms=9, mew=1.5)
    ax2.plot(*case["ftbtm"], "D", color=color, ms=6, mfc="none", mew=1.2)
    ax2.plot(*case["tests"], "s", color=color, ms=7, mfc="none", mew=1.2)
    ax2.plot(alphas, ltb / ltbeamn_u, "x", color=color, ms=7, mew=1.5)

cases_leg = [Line2D([], [], color=c["style"][0], ls=c["style"][1], lw=1.5, label=f"PyLTB, ({n})")
             for n, c in cases.items()]
gray = "#52514e"
kt = Line2D([], [], ls="", marker="+", color=gray, ms=9, mew=1.5, label="Kitipornchai and Trahair (1975)")
ltb_leg = Line2D([], [], ls="", marker="x", color=gray, mew=1.5, label="LTBeamN")
src_1 = [Line2D([], [], ls="", marker="o", color=gray, label="Bradford and Cuk (1988)"),
         Line2D([], [], ls="", marker="s", color=gray, label="Hancock and Trahair (1978)"), kt, ltb_leg]
src_2 = [kt, Line2D([], [], ls="", marker="D", color=gray, mfc="none", label="Trahair (2014), FTBTM"),
         Line2D([], [], ls="", marker="s", color=gray, mfc="none", label="Test (Trahair 2014)"), ltb_leg]
for ax, src, ylabel, ytop in ((ax1, src_1, r"$W_{cr}$ [N]", 1250), (ax2, src_2, r"$Q/Q_u$", 1.3)):
    ax.set_xlim(0, 1); ax.set_ylim(0, ytop)
    ax.set_xlabel(r"Flange taper constant $\alpha_B$"); ax.set_ylabel(ylabel)
    ax.grid(color="#d9d9d6", lw=0.6)
    ax.legend(handles=cases_leg + src, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=8, frameon=False)
ax1.set_title("Critical load (Bradford and Cuk 1988, Fig. 6)", fontsize=10)
ax2.set_title("Dimensionless load (Trahair 2014, Fig. 8)", fontsize=10)
fig.tight_layout()
plt.show()
