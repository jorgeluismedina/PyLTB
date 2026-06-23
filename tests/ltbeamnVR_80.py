
import numpy as np

import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=0.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_MS(h=0.5, bf1=0.28, bf2=0.28, tw=0.0125, tf1=0.019, tf2=0.019, r1=0.0, r2=0.0) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
L = 18 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 100
# Con 100 elementos mu_cr = 7.0401, error con Ansys delta = 0.98%
# Con 200 elementos mu_cr = 7.0396, error con Ansys delta = 0.97%
# Con 300 elememtos mu_cr = 7.0395, error con Ansys delta = 0.97%


# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)

# Generacion de secciones
sections = [sect1] * nodes.shape[0]

# Informacion de elementos
elements_data = np.array([[0, 0, e, e+1] for e in range(nelems)])



# ----- RESTRICCIONES --------
# Restricciones problema estatico
verax_restraints = np.array([
    [0,         1, 1, 0],
    [nelems,    1, 1, 0]
])

# restricciones problema de estabilidad
lator_restraints = np.array([
    [0,         1, 0, 1, 0],
    [nelems//2, 1, 0, 1, 0],
    [nelems,    1, 0, 1, 0]
])

# resortes laterales
#kpen = material1.E * sect1.Iy * 1e6
#springs_data = np.array([
#    [nelems//2, 0,  kpen, 0.0, kpen, 0.0]
#])


# ----- CARGAS DE ELEMENTO --------
# Cargas distribuida uniforme
elem_loads = []
for e in range(nelems):
    elem_loads.append([e, 0, 1,   0.0, 0.0,    0.0, -3000.0, 0.0, -3000.0])

elem_loads = np.array(elem_loads)




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_uniform_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
static.summary()
stabi.summary(ref={"Ansys": 6.9718, "LTBeamN": 6.9985})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.15)
plt.show()
