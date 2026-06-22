
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

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
sect1 = ISection_MS(h=0.3, bf1=0.20, bf2=0.15, tw=0.01, tf1=0.015, tf2=0.012, r1=0.0, r2=0.0) #[m]




# ----- CONSTRUCCION DE LA MALLA --------
L = 19.5 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 100 
# Con 150 elementos mu_cr = 9.2614, error con Ansys delta = 0.09%
# Con 250 elementos mu_cr = 9.2783, error con Ansys delta = 0.27%
# Con 400 elementos mu_cr = 9.3052, error con Ansys delta = 0.56%

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
    [nelems/2,  1, 1, 0],
    [nelems,    1, 1, 0]
])

# restricciones problema de estabilidad
lator_restraints = np.array([
    [0,         1, 0, 1, 0],
    [nelems/2,  1, 0, 1, 0],
    [nelems,    1, 0, 1, 0]
])



# ----- CARGAS NODALES --------
nodal_loads = np.array([
    [nelems/4, 0, 1,    0.0, 0.0,    0.0, -10000.0, 0.0]
])




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_uniform_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
static.summary()
stabi.summary(ref={"Ansys": 9.253, "LTBeamN": 9.2328})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.15)
plt.show()