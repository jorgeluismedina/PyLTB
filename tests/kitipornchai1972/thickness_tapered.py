import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# Kitipornchai y Trahair (1972), segun Bradford y Cuk (1988), Fig. 5: viga bisimetrica
# simplemente apoyada de 1524 mm con carga central en el ala superior. El espesor de las dos
# alas baja linealmente de T en el centro a alpha*T en los apoyos; la distancia entre
# centroides de alas es constante, de modo que el canto total es hc + T(x).


# ----- MATERIAL --------
materials = [Material(E=65160e6, nu=65160/(2*25650) - 1, rho=1.0)]   # G = 25650 MPa

# ----- SECCIONES --------
idx    = 0
alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
alpha  = alphas[idx]

hc, B, T, t = 0.0728, 0.0316, 0.00311, 0.00213   # [m], hc entre centroides de alas
section_max = ISection_MS(h=hc + T, bf1=B, bf2=B, tw=t, tf1=T, tf2=T, r1=0.00, r2=0.00, It_type="plates")
section_min = ISection_MS(h=hc + alpha*T, bf1=B, bf2=B, tw=t, tf1=alpha*T, tf2=alpha*T, r1=0.00, r2=0.00, It_type="plates")


# ----- CONSTRUCCION DE LA MALLA --------
L      = 1.524 #[m]
nelems = 20    # par: el nodo nelems//2 queda en el centro
nnods  = nelems + 1

# Coordenadas de nodos
nodes = np.linspace(0, L, nelems+1)
norm_coords = nodes / L

# Mitad izquierda de 0.0 a 1.0 y mitad derecha de 0.0 a 1.0
norm_coords_left  = norm_coords[0:nnods//2+1] * 2.0
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
# Apoyos de horquilla
lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0],
])


# ----- CARGAS NODALES --------
# Carga central de 1 N en el centroide del ala superior (T/2 bajo la fibra superior),
# porque Bradford y Cuk miden h entre centroides de alas
nodal_loads = np.array([
    [nelems//2,   0, 3,    0.0, -T/2,    0.0, -1.0, 0.0]
])


# ----- CREACION Y SETEO DEL MODELO --------
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)
model.summary()


# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
# W_cr [N]. Ref.: puntos de Bradford y Cuk (1988), Fig. 5, leidos del grafico (±1 %).
# LTBeamN 2.0.2: media viga con simetria, It de Villette. Con alpha <= 0.7 LTBeamN toma
# I_wpsi = dIw/dx, que no es nulo aunque la distancia entre centroides de alas sea constante,
# y su carga critica cae hasta 58 %; con alpha = 0.1 no resuelve (codigo de error 56).
mu_cr_ref     = [520, 543, 574, 607, 652, 706, 769, 832, 906]
mu_cr_ltbeamn = [np.nan, 229.3, 395.5, 492.1, 572.7, 649.0, 725.2, 826.3, 893.6]
static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx],
                   "LTbeamN": mu_cr_ltbeamn[idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.04)
plt.show()
