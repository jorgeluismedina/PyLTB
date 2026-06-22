
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_bs import ISection_BS
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.15, tw=0.015, tf=0.015, r=0.01) #[m]
#sect1.summary()



# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 20 #50

# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)

# Generacion de secciones
sections = [sect1] * nodes.shape[0]


# Informacion de elementos
elements_data = np.array([[0, 0, e, e+1] for e in range(nelems)])



# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  0, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])



# ----- CARGAS DE ELEMENTO --------
# Carga distribuida uniforme unitaria
elem_loads = np.array([[e, 0, 1,  0.0, 0.0,  0.0, -1e3, 0.0,-1e3] for e in range(nelems)])




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
stabi.summary(ref={ "LTbeamN": 82.41})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.15)
plt.show()