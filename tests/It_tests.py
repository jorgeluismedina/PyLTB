
"""
Test  –  Inercia de torsion It de secciones en I monosimetricas
  Las tres formulas se toman directamente de ISection_MS (PyLTB), no se
  reimplementan aqui:
    Darwish  : ISection_MS.It_darwish()   – Darwish & Johnston (1965)
    Villette : ISection_MS.It_villette()  – Villette (2011), la que usa LTBeamN
    Plates   : ISection_MS.It_plates()    – suma simple de placas (default de PyLTB)
Unidades: dimensiones en m, resultados en cm⁴.
"""

from pyltb.sections.section_ms import ISection_MS


# ── secciones: (h, bf1, bf2, tw, tf1, tf2, r1, r2) [m] ─────────────────────────

SECTIONS = {
    "Ej1 BS  h=610":   (0.610,   0.18, 0.18, 0.0080, 0.0100, 0.0100, 0.000, 0.000),
    "Ej1 BS  h=305":   (0.305,   0.18, 0.18, 0.0080, 0.0100, 0.0100, 0.000, 0.000),
    "Ej1 MS  h=610":   (0.610,   0.10, 0.18, 0.0080, 0.0100, 0.0100, 0.000, 0.000),
    "Ej1 MS  h=305":   (0.305,   0.10, 0.18, 0.0080, 0.0100, 0.0100, 0.000, 0.000),
    "Ej2     h=612.7": (0.6127,  0.15, 0.15, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "Ej2     h=122.5": (0.12254, 0.15, 0.15, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "Ej3/5   h=600":   (0.600,   0.15, 0.15, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "Ej3/5   h=240":   (0.240,   0.15, 0.15, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "Ej4     h=600":   (0.600,   0.20, 0.05, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "Ej4     h=240":   (0.240,   0.20, 0.05, 0.0095, 0.0127, 0.0127, 0.000, 0.000),
    "IPE 300":         (0.300,   0.15, 0.15, 0.0071, 0.0107, 0.0107, 0.015, 0.015),
    "IPE 500":         (0.500,   0.20, 0.20, 0.0102, 0.0160, 0.0160, 0.021, 0.021),
}

# valores de LTBeamN [cm⁴]  (None → sin dato)
LTBEAMN = {
    "Ej1 BS  h=610":   21.9,
    "Ej1 BS  h=305":   16.7,
    "Ej1 MS  h=610":   19.23,
    "Ej1 MS  h=305":   14.03,
    "Ej2     h=612.7": 36.72,
    "Ej2     h=122.5": 22.71,
    "Ej3/5   h=600":   36.36,
    "Ej3/5   h=240":   26.07,
    "Ej4     h=600":   32.94,
    "Ej4     h=240":   22.65,
    "IPE 300":         19.87,
    "IPE 500":         89.01,
}


# ── run ────────────────────────────────────────────────────────────────────────

M4_TO_CM4 = 1e8


def col(x):
    return f"{x:>12.3f}" if x is not None else f"{'—':>12}"


def delta(x, ref):
    return f"{100 * (x / ref - 1):>+10.2f}%" if ref else f"{'—':>11}"


print("\n" + "═" * 100)
print("  Inercia de torsion It [cm⁴]  –  Δ respecto a LTBeamN")
print("═" * 100)
print(f"  {'Seccion':<17}{'LTBeamN':>12}{'Darwish':>12}{'ΔLTBeamN':>11}"
      f"{'Villette':>12}{'ΔLTBeamN':>11}{'Plates':>12}{'ΔLTBeamN':>11}")
print("  " + "─" * 96)

for name, dims in SECTIONS.items():
    sec = ISection_MS(*dims)
    da  = sec.It_darwish()  * M4_TO_CM4
    vi  = sec.It_villette() * M4_TO_CM4
    pl  = sec.It_plates()   * M4_TO_CM4
    ltb = LTBEAMN[name]
    print(f"  {name:<17}{col(ltb)}"
          f"{col(da)}{delta(da, ltb)}"
          f"{col(vi)}{delta(vi, ltb)}"
          f"{col(pl)}{delta(pl, ltb)}")

print("═" * 100 + "\n")
