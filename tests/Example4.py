
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.sections.section_utils import interpolate_multiple_sections
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import (
    plot_diagrams,
    plot_buckling_mode,
)

# ----- MATERIAL --------
material1 = Material(E=2.10e11, nu=0.3, dens=1.0)
materials = [material1]

# ----- SECCIONES --------
section_max = ISection_MS(h=0.60, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) #[m]
L   = Ls[idx]

nelems = int(10 * L / 2)
nnods  = nelems + 1

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Re-escalar para que la mitad izquierda vaya de 0.0 a 1.0
norm_coords_left = norm_coords[0:nnods//2+1] * 2.0

# Re-escalar para que la mitad derecha vaya de 0.0 a 1.0
# (restamos 0.5 para que empiece en 0, y multiplicamos por 2)
norm_coords_right = (norm_coords[nnods//2+1:] - 0.5) * 2.0

# Generacion de secciones
node_sections_left  = interpolate_multiple_sections(section_min, section_max, norm_coords_left)
node_sections_right = interpolate_multiple_sections(section_max, section_min, norm_coords_right)
node_sections = node_sections_left + node_sections_right

# Informacion de elementos
elements_data = []
for e in range(nelems):
    # formato: [etype, mat_id, nodei, nodej]
    elements_data.append([1, 0, e, e+1]) 

elements_data = np.array(elements_data)


# ----- RESTRICCIONES --------
# Simplemente apoyado
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  0, 1, 0],
])
# Simplemente apoyado
lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0],
])


# ----- CARGAS NODALES --------
# Aproximando centro de torsion

# 1. Calcular las coordenadas locales Z respecto al centroide (align = 0)
# z_from_ref(align=0, pos=1) da la distancia del Centroide (0) al SC (1)
z_SC_apoyo = section_min.z_from_ref(0, 1)  # Esta es la constante para la línea TC
z_SC_centr = section_max.z_from_ref(0, 1)  # SC local que usa LTBeamN por defecto

# 2. La distancia exacta a sumar
rez_exacto = np.abs(z_SC_apoyo - z_SC_centr)
print(rez_exacto)


nodal_loads = np.array([
    #[nelems//2,    0, 3,    0.0, 0.0,    0.0, -1000.0, 0.0] # solo sobre la mesa superior
    [nelems//2,    0, 1,    0.0, -rez_exacto,    0.0, -1000.0, 0.0] # con excentricidad artificial
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
mu_cr_ref     = [85.09, 39.27, 22.47]
mu_cr_ltbeamn = [57.94, 29.71, 18.14]


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
print(f"  Lenght (L):                             {L:>11.2f} m")
print(f"  Critical load factor μ_cr (PyLTB):      {mu_cr:>12.4f}")
print(f"  Critical load factor μ_cr (Reference):  {mu_cr_ref[idx]:>12.4f}")
print(f"  Critical load factor μ_cr (LTBeamN):    {mu_cr_ltbeamn[idx]:>12.4f}")
print(f"  Result diff. with Reference:            {abs(mu_cr - mu_cr_ref[idx])/mu_cr_ref[idx]*100:>11.2f} %")
print(f"  Result diff. with LTBeamN:              {abs(mu_cr - mu_cr_ltbeamn[idx])/mu_cr_ltbeamn[idx]*100:>11.2f} %")
print("\n" + "="*55 + "\n")



#"""
# ----- PLOTEO DE RESULTADOS --------
plot_diagrams(model, static.diagrams, static.deformations)
plot_buckling_mode(model, stabi.mu_crs, stabi.modes_SC, imode=0, scale=0.15, n_sec=2)
plt.show()
#"""