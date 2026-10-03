import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# Beyer et al. (2015), Example #4: viga monosimetrica con doble ahusamiento, secciones alineadas
# en el centroide (el eje de centros de corte tiene un quiebre en el centro), carga central
# en la mesa superior.


# ----- MATERIAL --------
material1 = Material(E=2.10e11, nu=0.3, rho=1.0)
materials = [material1]

# ----- SECCIONES --------
section_max = ISection_MS(h=0.60, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00, It_type="plates") #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.20, bf2=0.05, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00, It_type="plates") #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) #[m]
L   = Ls[idx]

nelems = int(4 * L)   # par: el nodo nelems//2 queda en el centro
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
# e(SC): altura de la carga medida desde el centro de corte local (lo que hace el programa).
# e(TC): altura medida desde la linea de centros de torsion, la recta por los centros de corte
#        de los apoyos (Beyer 2015, §4.5). Se emula bajando la carga rez desde la mesa superior.
# z_from_ref(align=0, pos=1) da la distancia del centroide (0) al SC (1); el eje de centroides
# es recto, asi que la diferencia es la distancia entre el SC del centro y la linea TC.
z_SC_apoyo = section_min.z_from_ref(0, 1)  # constante de la linea TC
z_SC_centr = section_max.z_from_ref(0, 1)  # SC local en el centro
rez = np.abs(z_SC_apoyo - z_SC_centr)

nodal_loads = {
    "e(SC)": np.array([[nelems//2, 0, 3,    0.0, 0.0,    0.0, -1000.0, 0.0]]),  # sobre la mesa superior
    "e(TC)": np.array([[nelems//2, 0, 3,    0.0, -rez,   0.0, -1000.0, 0.0]]),  # bajar rez desde la mesa superior
}


# ----- CREACION Y SETEO DEL MODELO --------
def build_model(loads):
    model = StabilityModel()
    model.add_materials(materials)
    model.add_sections(sections)
    model.add_nodes(nodes)
    model.add_tapered_elements(elements_data)
    model.add_verax_restraints(verax_restraints)
    model.add_lator_restraints(lator_restraints)
    model.add_nodal_loads(loads)
    return model


# ── Resolver ──────────────────────────────────────────────────────────────
results = {}
for key, loads in nodal_loads.items():
    model  = build_model(loads)
    static = StaticSolver(model).solve()
    stabi  = StabilitySolver(model).solve()
    results[key] = (model, static, stabi)

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_ref     = [85.09, 39.27, 22.47]             # Ansys (Beyer 2015, Table 5)
mu_cr_ltbeamn = {"e(TC)": [80.56, 37.52, 21.60],  # LTBeamN, programa
                 "e(SC)": [56.20, 28.99, 17.69]}
# Articulo (Table 5): e(TC) = [83.73, 38.71, 22.18], e(SC) = [57.94, 29.71, 18.14]. Esos valores
# corresponden a la carga en el centro de corte; con la carga en la mesa superior se obtienen
# los del programa.
results["e(SC)"][0].summary()   # modelo
print(f"rez = {rez:.4f} m")
results["e(SC)"][1].summary()   # problema estático
for key, (_, static, stabi) in results.items():
    print(f"Altura de la carga medida con {key}")
    stabi.summary(ref={"Ref.": mu_cr_ref[idx],
                       f"LTBeamN {key}": mu_cr_ltbeamn[key][idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
results["e(SC)"][1].plot()
for key, (_, static, stabi) in results.items():
    fig, _ = stabi.plot(imode=0, scale=0.20)
    fig.suptitle(key)
plt.show()
