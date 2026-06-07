
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

# Secciones
section1 = ISection_MS(h=0.61, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]
section2 = ISection_MS(h=0.305, bf1=0.18, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]


# ----- CONSTRUCCION DE LA MALLA --------
idx = 1                        # índice de longitud a analizar
Ls  = np.array([2, 4, 6, 8, 10])
L   = Ls[idx]

nelems = int(16 * L / 2)
nnods  = nelems + 1

# Coordenadas de nodos
coordinates  = np.linspace(0, L, nnods)
norm_coords  = coordinates / L

# Generacion de secciones
node_sections = interpolate_multiple_sections(section1, section2, norm_coords)


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


# ----- RESTRICCIONES --------
# Empotramiento
verax_restraints = np.array([
    [0,       1, 1, 1],
])
# Empotramiento
lator_restraints = np.array([
    [0,       1, 1, 1, 1],
])


# ---------- CARGA ----------
# Carga puntual vertical Q1 en el extremo libre hacia abajo
# Carga puntual axial Q2 en el extremo libre hacia la izquierda
# sobre el ala superior → pos=3
# sobre el ala superior → pos=3
nodal_loads = np.array([
    #[nelems,  3, 3,   0.0, 0.0,   -1e3, -1e3, 0.0]  
    [nelems,  3, 3,   0.0, 0.0,   -10e3, -5e3, 0.0]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data, align=0)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)



# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
static = StaticSolver(model)
static.solve()
maxN, maxV, maxM, maxw = static.max_vals() 

print(model.elements[0].forcesG)
print(model.elements[0].forces)

# Resolcion del problema de estabilidad
stabi = StabilitySolver(model)
stabi.solve()
mu_cr = stabi.mu_crs[0]


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
#'''
print("\n STABILITY ANALYSIS")
print(f"  Lenght (L):                             {L:>11.2f} m")
print(f"  Critical load factor μ_cr (PyLTB):      {mu_cr:>12.4f}")
#print(f"  Critical load factor μ_cr (Reference):  {mu_cr_ref[idx]:>12.4f}")
#print(f"  Critical load factor μ_cr (LTBeamN):    {mu_cr_ltbeamn[idx]:>12.4f}")
#print(f"  Result diff. with Reference:            {abs(mu_cr - mu_cr_ref[idx])/mu_cr_ref[idx]*100:>11.2f} %")
#print(f"  Result diff. with LTBeamN:              {abs(mu_cr - mu_cr_ltbeamn[idx])/mu_cr_ltbeamn[idx]*100:>11.2f} %")
print("\n" + "="*55 + "\n")
#'''



#"""
# ----- PLOTEO DE RESULTADOS --------
# Problema estatico
N_diag, V_diag, M_diag, def_shapes = static.prepare_diagrams()
 
plot_diagram(model, N_diag,    title="Axial force")
plot_diagram(model, V_diag,    title="Shear force")
plot_diagram(model, M_diag,    title="Bending moment")
#plot_deformed(model, def_shapes, title="Deformed shape")

# Problema de estabilid
#plot_buckling_modes(model, stabi.mu_crs, stabi.modes, nmodes=2)
#plot_buckling_mode_3d(model, stabi.mu_crs, stabi.modes, imode=0, scale=0.015, n_sec=5)

plt.show()
#"""