
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import build_mesh 
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# ── Materiales ────────────────────────────────────────────────────────────
materials = [Material(E=2.1e11, nu=0.3, rho=1.0)]

# ── Secciones ─────────────────────────────────────────────────────────────
section1 = ISection_MS(h=0.61, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]
section2 = ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]
section_order = [section1, section2]
section_breakpoints = [0.0, 1.0]
#section2.summary()

# ── Malla ─────────────────────────────────────────────────────────────────
idx = 0 # índice de longitud a analizar
Ls  = np.array([2, 4, 6, 8, 10])
nelems = int(4*Ls[idx])

nodes, sections, elements_data = build_mesh(Ls[idx], section_breakpoints, section_order, 
                                            nelems, etype=1, mat_id=0)

# ── Modelo ────────────────────────────────────────────────────────────────
# Empotramiento
verax_restraints = np.array([[0,  1, 1, 1]])
lator_restraints = np.array([[0,  1, 1, 1, 1]])

# Carga vertical Q en el extremo libre sobre el ala superior
nodal_loads = np.array([
    [nelems,  0, 3,   0.0, 0.0,   0.0, -1000.0, 0.0]
])

model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_tapered_elements(elements_data, align=3)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()


# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_ref     = [173.30, 44.55, 22.69, 13.95, 9.31]
mu_cr_ltbeamn = [176.50, 45.10, 22.83, 13.97, 9.30] # del articulo de Beyer, 2015
mu_cr_ltbeamn = [171.87, 44.23, 22.50, 13.82, 9.22] # con el programa
static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx], 
                   "LTbeamN": mu_cr_ltbeamn[idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.015)
plt.show()
