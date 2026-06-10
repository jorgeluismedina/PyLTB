
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.model import StabilityModel
from src.material import Material
from src.sections.section_bs import ISection_BS
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
#from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed
from src.plotting import (
    plot_diagrams,
    plot_buckling_mode,
)

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.15, tw=0.015, tf=0.015, r=0.01) #[m]
sect1.summary()


# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 10 

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)

# Generacion de secciones
node_sections = [sect1] * coordinates.shape[0]


# Informacion de elementos
elements_data = []

for e in range(nelems):
    # formato: [etype, mat_id, nodei, nodej]
    elements_data.append([0, 0, e, e+1])
elements_data = np.array(elements_data)



# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  0, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])


# ----- CARGAS NODALES --------
# Carga de flexion pura unitaria
nodal_loads = np.array([
    [0,      0, 1,   0.0, 0.0,   0.0, 0.0, -1000.0],
    [nelems, 0, 1,   0.0, 0.0,   0.0, 0.0,  1000.0]
])





# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_uniform_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)


# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
static = StaticSolver(model)
static.solve()
maxN, maxV, maxM, maxw = static.max_vals() 

# Resolcion del problema de estabilidad
stabi = StabilitySolver(model)
stabi.solve()
mu_cr = stabi.mu_crs[0]


# Resultados y comparacion
EIz = material1.E * sect1.Iz
GIt = material1.G * sect1.It
EIw = material1.E * sect1.Iw

mu_cr_ana = np.pi / L * np.sqrt(EIz*GIt * (1 + (np.pi**2*EIw)/(L**2*GIt))) / 1000 
mu_cr_ltbeamn = 228.1


print("\n" + "="*55)
print(" ANALYSIS RESULTS ".center(55))
print("="*55)

print("\n MESH DATA")
print(f"  Number of nodes:                 {model.nnodes:>20}")
print(f"  Number of elements:              {model.nelems:>20}")

print("\n STATIC ANALYSIS")
print(f"  Axial max.        Nmax:          {maxN/1e3:>16.4f} kN")
print(f"  Shear max.        Vmax:          {maxV/1e3:>16.4f} kN")
print(f"  Moment max.       Mmax:          {maxM/1e3:>16.4f} kNm")
print(f"  Displacement max. w_max:         {maxw*1e3:>16.4f} mm")

print("\n STABILITY ANALYSIS")
print(f"  Critical load factor μ_cr (Real):       {mu_cr_ana:>12.4f}")
print(f"  Critical load factor μ_cr (PyLTB):      {mu_cr:>12.4f}")
print(f"  Critical load factor μ_cr (LTBeamN):    {mu_cr_ltbeamn:>12.4f}")
print(f"  Error respect analytical:               {abs(mu_cr - mu_cr_ana)/mu_cr_ana*100:>11.2f} %")
print(f"  Result diff. with LTBeamN:              {abs(mu_cr - mu_cr_ltbeamn)/mu_cr_ltbeamn*100:>11.2f} %")
print("\n" + "="*55 + "\n")
 

#"""
# ----- PLOTEO DE RESULTADOS --------
plot_diagrams(model, static.diagrams, static.deformations)
plot_buckling_mode(model, stabi.mu_crs, stabi.modes_SC, imode=0, scale=0.15, n_sec=2)
plt.show()
#"""
