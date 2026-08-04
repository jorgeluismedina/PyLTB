

import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# Materiales
material1 = Material(E=2.10e11, nu=0.3, rho=1.0)
materials = [material1]

# Secciones
section_max = ISection_MS(h=0.60, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) / 2 #[m]
L   = Ls[idx]

nelems = int(10 * L)
nnods  = nelems + 1

# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)
norm_coords = nodes / L

# Generacion de secciones
sections = interpolate_multiple_sections(section_min, section_max, norm_coords)


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       0, 1, 0],
    [nelems,  1, 0, 1]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  0, 1, 0, 1]
])


# ----- CARGAS NODALES --------
# Carga puntual en la punta sobre la mesa superior
nodal_loads = np.array([
    [nelems, 0, 3,   0.0, 0.0,   0.0, -500.0, 0.0]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_ref     = np.array([63.58, 30.55, 18.05])
mu_cr_ltbeamn = np.array([62.17, 29.97, 17.76])
mu_cr_ltbeamn = np.array([61.20, 29.65, 17.60])
static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx], 
                   "LTbeamN": mu_cr_ltbeamn[idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.1)
plt.show()
