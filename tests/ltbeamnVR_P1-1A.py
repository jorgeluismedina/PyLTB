
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
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2] 
materials = [material1]

# Secciones
section1 = ISection_MS(h=0.44, bf1=0.25, bf2=0.25, tw=0.014, tf1=0.02, tf2=0.02, r1=0.0, r2=0.0) #[m]
section2 = ISection_MS(h=0.84, bf1=0.25, bf2=0.25, tw=0.014, tf1=0.02, tf2=0.02, r1=0.0, r2=0.0) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 100 

# Coordenada de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Generacion de secciones
node_sections = interpolate_multiple_sections(section1, section2, norm_coords)


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  1, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])


# ----- CARGAS NODALES --------
# Carga de flexion pura unitaria
nodal_loads = np.array([
    [0,      0,   0.0, 0.0,    0.0, 0.0,  200000.0],
    [nelems, 0,   0.0, 0.0,    0.0, 0.0, -800000.0]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
static.summary()
stabi.summary(ref={"Ansys": 4.7321, "LTBeamN": 3.162})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.15)
plt.show()