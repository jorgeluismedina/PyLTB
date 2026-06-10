
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import (
    plot_diagrams,
    plot_buckling_mode,
)

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=0.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_MS(h=0.5, bf1=0.28, bf2=0.28, tw=0.0125, tf1=0.019, tf2=0.019, r1=0.0, r2=0.0) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
L = 18 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 100


# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)

# Generacion de secciones
node_sections = [sect1] * coordinates.shape[0]

# Informacion de elementos
elements_data = []
for e in range(nelems):
    elements_data.append([0, 0, e, e+1]) # etype, mat_id, sec_id, nodei, nodej
elements_data = np.array(elements_data)



# ----- RESTRICCIONES --------
# Restricciones problema estatico
verax_restraints = np.array([
    [0,         1, 1, 0],
    [nelems,    1, 1, 0]
])

# restricciones problema de estabilidad
lator_restraints = np.array([
    [0,         1, 0, 1, 0],
    [nelems//4, 1, 0, 1, 0],
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
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_uniform_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)


# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
static = StaticSolver(model)
static.solve()
maxN, maxV, maxM, maxw = static.max_vals()

# Resolcion del problema de estabilidad
stabi = StabilitySolver(model)
stabi.solve()
mu_cr = stabi.mu_crs[0]

mu_cr_ansys = 5.1208
mu_cr_ltbeamn = 5.1275


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
print(f"  Critical load factor μ_cr (PyLTB):      {mu_cr:>12.4f}")
print(f"  Critical load factor μ_cr (Ansys):      {mu_cr_ansys:>12.4f}")
print(f"  Critical load factor μ_cr (LTBeamN):    {mu_cr_ltbeamn:>12.4f}")
print(f"  Error respect Ansys:                    {abs(mu_cr - mu_cr_ansys)/mu_cr_ansys*100:>11.2f} %")
print(f"  Result diff. with LTBeamN:              {abs(mu_cr - mu_cr_ltbeamn)/mu_cr_ltbeamn*100:>11.2f} %")
print("\n" + "="*55 + "\n")

 

#"""
# ----- PLOTEO DE RESULTADOS --------
plot_diagrams(model, static.diagrams, static.deformations)
plot_buckling_mode(model, stabi.mu_crs, stabi.modes_SC, imode=0, scale=0.01, n_sec=2)
plt.show()
#"""