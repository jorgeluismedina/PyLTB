
import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# Beyer et al. (2015), Example #5, medio modelo con condiciones de simetria en el centro.
# Carga central en el centro de corte (P/2 en el nodo de simetria).


# Materiales
material1 = Material(E=2.10e11, nu=0.3, rho=1.0)
materials = [material1]

# ----- SECCIONES --------
section_max = ISection_MS(h=0.60, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00, It_type="plates") #[m]
section_min = ISection_MS(h=0.60*0.4, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00, It_type="plates") #[m]



# ----- CONSTRUCCION DE LA MALLA --------
idx = 0
Ls  = np.array([6, 9, 12]) / 2 #[m]
L   = Ls[idx]

nelems = int(4 * L)
nnods  = nelems + 1

# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)
norm_coords = nodes / L

# Generacion de secciones
sections = interpolate_multiple_sections(section_min, section_max, norm_coords)


# Informacion de elementos
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


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
# e(SC): altura de la carga medida desde el centro de corte local (lo que hace el programa).
# e(TC): altura medida desde la linea de centros de torsion, la recta por los centros de corte
#        de los apoyos (Beyer 2015, §4.6). Se emula subiendo la carga rez desde el centro de corte.
# z_from_ref(align=3, pos=1) da la distancia de la mesa superior (3) al SC (1); la mesa superior
# es recta, asi que la diferencia es la distancia entre el SC del centro y la linea TC.
z_SC_apoyo = section_min.z_from_ref(3, 1)  # constante de la linea TC
z_SC_centr = section_max.z_from_ref(3, 1)  # SC local en el centro
rez = np.abs(z_SC_apoyo - z_SC_centr)

nodal_loads = {
    "e(SC)": np.array([[nelems, 0, 1,   0.0, 0.0,   0.0, -500.0, 0.0]]),  # en el centro de corte
    "e(TC)": np.array([[nelems, 0, 1,   0.0, rez,   0.0, -500.0, 0.0]]),  # subir rez desde el SC
}


# ----- CREACION Y SETEO DEL MODELO --------
def build_model(loads):
    model = StabilityModel()
    model.add_materials(materials)
    model.add_sections(sections)
    model.add_nodes(nodes)
    model.add_tapered_elements(elements_data, align=3)
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
mu_cr_ref     = [146.40, 56.20, 29.23]              # Ansys (Beyer 2015, Table 6)
mu_cr_ltbeamn = {"e(TC)": [143.93, 55.41, 28.84],   # LTBeamN, programa
                 "e(SC)": [192.44, 68.97, 34.39]}
# Articulo (Table 6): e(TC) = [147.50, 56.25, 29.19], e(SC) = [193.70, 69.32, 34.55]
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
    fig, _ = stabi.plot(imode=0, scale=0.15)
    fig.suptitle(key)
plt.show()
