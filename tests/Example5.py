
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver
# ----- MATERIAL --------
material1 = Material(E=2.10e11, nu=0.3, rho=1.0)
materials = [material1]

# ----- SECCIONES --------
section_max = ISection_MS(h=0.60, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) #[m]
L   = Ls[idx]

nelems = int(10 * L / 2)
nnods  = nelems + 1

# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)
norm_coords = nodes / L

# Re-escalar para que la mitad izquierda vaya de 0.0 a 1.0
norm_coords_left = norm_coords[0:nnods//2+1] * 2.0

# Re-escalar para que la mitad derecha vaya de 0.0 a 1.0
# (restamos 0.5 para que empiece en 0, y multiplicamos por 2)
norm_coords_right = (norm_coords[nnods//2+1:] - 0.5) * 2.0


# Generacion de secciones
sections_left  = interpolate_multiple_sections(section_min, section_max, norm_coords_left)
sections_right = interpolate_multiple_sections(section_max, section_min, norm_coords_right)
sections = sections_left + sections_right


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


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
# Carga puntual en el centro sobre el centro de corte
# 1. Calcular las coordenadas locales Z respecto al centroide (align = 0)
# z_from_ref(align=0, pos=1) da la distancia del Centroide (0) al SC (1)
z_SC_apoyo = section_min.z_from_ref(3, 1)  # Esta es la constante para la línea TC
z_SC_centr = section_max.z_from_ref(3, 1)  # SC local que usa LTBeamN por defecto

# 2. La distancia exacta a restar
rez_exacto = np.abs(z_SC_apoyo - z_SC_centr)
print(rez_exacto)

nodal_loads = np.array([
    #[nelems//2, 0, 1,   0.0, 0.0,   0.0, -1000.0, 0.0]
    [nelems//2, 0, 1,   0.0, rez_exacto,   0.0, -1000.0, 0.0]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_tapered_elements(elements_data, align=3)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_ref     = [146.4, 56.20, 29.23]
mu_cr_ltbeamn = [193.7, 69.32, 34.55]
static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx], 
                   "LTbeamN": mu_cr_ltbeamn[idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.18)
plt.show()
