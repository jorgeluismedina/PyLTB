"""
Ejemplo 1 de las normas: ménsula I soldada bisimétrica de alma variable, carga puntual en la punta
sobre la cara superior de la mesa superior. Desarrollo NBR 8800:2024 en
D:\\Maestria UFRGS\\Tesis Maestria\\Disertacion\\_bibliografia\\bib_aprobada\\Normas\\Ejemplos\\Ejemplo1_mensula\\NBR

  mesas 200 x 16 mm (constantes), alma 9,5 mm, altura total 450 mm (raíz) -> 225 mm (punta), L = 4 m
  E = 200 GPa, G = 77 GPa (NBR 4.6.2.9), P_Sd = 90 kN  ->  μ_cr = α_cr = P_cr / P_Sd
  mesa superior recta (align=3); soldada sin radios: It como suma de placas, igual que J en el .md
"""
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── Material ──────────────────────────────────────────────────────────────
E, G = 200e9, 77e9
materials = [Material(E=E, nu=E/(2*G) - 1, rho=1.0)]   # G = 77 GPa

# ── Secciones ─────────────────────────────────────────────────────────────
section_root = ISection_MS(h=0.450, bf1=0.200, bf2=0.200, tw=0.0095, tf1=0.016, tf2=0.016, r1=0.0, r2=0.0, It_type="plates") #[m]
section_tip  = ISection_MS(h=0.225, bf1=0.200, bf2=0.200, tw=0.0095, tf1=0.016, tf2=0.016, r1=0.0, r2=0.0, It_type="plates") #[m]

# ── Malla ─────────────────────────────────────────────────────────────────
L      = 4.0 #[m]
nelems = 32
coordinates   = np.linspace(0, L, nelems + 1)
node_sections = interpolate_multiple_sections(section_root, section_tip, coordinates / L)
elements_data = np.array([[1, 0, e, e + 1] for e in range(nelems)])

# ── Modelo ────────────────────────────────────────────────────────────────
# Empotramiento: v, v,x, θ y θ,x (alabeo) impedidos; punta libre
verax_restraints = np.array([[0,  1, 1, 1]])
lator_restraints = np.array([[0,  1, 1, 1, 1]])

# P_Sd = 90 kN hacia abajo en la punta, sobre la cara superior de la mesa superior
nodal_loads = np.array([
    [nelems,  0, 3,   0.0, 0.0,   0.0, -90e3, 0.0]
])

model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data, align=3)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_lba = 136.30 / 90   # lba_mensula.py (Normas/Herramientas): P_cr = 136,30 kN
static.summary()
stabi.summary(ref={"lba_mensula": mu_cr_lba})
print(f"P_cr = {stabi.mu_crs[0]*90:.2f} kN ; M_cr = {stabi.mu_crs[0]*90*L:.2f} kN·m (raíz)")

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.020)
plt.show()
