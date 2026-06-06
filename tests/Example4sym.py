
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
    plot_diagram,
    plot_deformed,
    plot_buckling_modes,
    plot_buckling_mode_3d,
)

# Materiales
material1 = Material(E=2.10e11, nu=0.3, dens=1.0)
materials = [material1]

# ----- SECCIONES --------
section_max = ISection_MS(h=0.60, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) / 2 #[m]
L   = Ls[idx]

nelems = int(10 * L)
nnods  = nelems + 1

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Generacion de secciones
node_sections = interpolate_multiple_sections(section_min, section_max, norm_coords)
#node_sections = interpolate_multiple_sections(section_max, section_min, norm_coords)


# Informacion de elementos
elements_data = []
for e in range(nelems):
    # formato: [etype, mat_id, nodei, nodej]
    elements_data.append([1, 0, e, e+1]) 

elements_data = np.array(elements_data)


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
# Aproximando centro de torsion

# 1. Calcular las coordenadas locales Z respecto al centroide (align = 0)
# z_from_ref(align=0, pos=1) da la distancia del Centroide (0) al SC (1)
z_SC_apoyo = section_min.z_from_ref(0, 1)  # Esta es la constante para la línea TC
z_SC_centr = section_max.z_from_ref(0, 1)  # SC local que usa LTBeamN por defecto

# 2. La distancia exacta a sumar
rez_exacto = np.abs(z_SC_apoyo - z_SC_centr)
print(rez_exacto)

nodal_loads = np.array([
    [nelems, 0, 3,    0.0, 0.0,    0.0, -500.0, 0.0] # solo sobre la mesa superior
    #[nelems, 0, 3,    0.0, rez_exacto,    0.0, -500.0, 0.0] # con excentricidad adicional
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
mu_cr_ref     = np.array([85.09, 39.27, 22.47])
mu_cr_ltbeamn = np.array([57.94, 29.71, 18.14])
mu_cr_ltbeamn = np.array([56.20, 28.99, 19.69])


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
# Problema estatico
N_diag, V_diag, M_diag, def_shapes = static.prepare_diagrams()
 
plot_diagram(model, N_diag,    title="Axial force")
plot_diagram(model, V_diag,    title="Shear force")
plot_diagram(model, M_diag,    title="Bending moment")
plot_deformed(model, def_shapes, title="Deformed shape")

# Problema de estabilidad
plot_buckling_modes(model, stabi.mu_crs, stabi.modes, nmodes=2)
plot_buckling_mode_3d(model, stabi.mu_crs, stabi.modes, imode=0, scale=0.12, n_sec=2)

plt.show()
#"""