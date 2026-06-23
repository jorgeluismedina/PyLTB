
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2] # cambio a nu=0.3 por que LTBeamN no me deja cambiar a 0.2
materials = [material1]

# Secciones
section1 = ISection_MS(h=0.3, bf1=0.20, bf2=0.15, tw=0.01, tf1=0.015, tf2=0.015, r1=0.01, r2=0.01) #[m]
section2 = ISection_MS(h=0.2, bf1=0.15, bf2=0.10, tw=0.01, tf1=0.015, tf2=0.015, r1=0.01, r2=0.01) #[m]

# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 20 

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Generacion de secciones
node_sections = interpolate_multiple_sections(section1, section2, norm_coords)


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0], # u=0, w=0, w,x=libre
    [nelems,  0, 1, 0]  # u=libre, w=0, w,x=libre
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0], # v=0, v,x=libre, theta=0, theta,x=libre
    [nelems,  1, 0, 1, 0]  # v=0, v,x=libre, theta=0, theta,x=libre
])


# ----- CARGAS DE ELEMENTO --------
# Carga distribuida uniforme unitaria
elem_loads = np.array([[e, 0, 1,  0.0, 0.0,  5e3, -1e3, 5e3,-1e3] for e in range(nelems)])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
static.summary()
stabi.summary(ref={ "LTbeamN": 133.9})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.15)
plt.show()
