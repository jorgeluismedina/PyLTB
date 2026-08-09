

import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver

# ── Materiales ────────────────────────────────────────────────────────────
material1 = Material(E=2.10e11, nu=0.3, rho=1.0)
materials = [material1]

# ── Secciones ─────────────────────────────────────────────────────────────
section1 = ISection_MS(h=0.6127, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]
section2 = ISection_MS(h=0.6127*0.2, bf1=0.15, bf2=0.15, tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]


# ── Malla ─────────────────────────────────────────────────────────────────
L = 4 #[m]
nelems = 20

nodes = np.linspace(0, L, nelems+1)
sections = interpolate_multiple_sections(section1, section2, nodes / L)

elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])


# ── Restricciones ─────────────────────────────────────────────────────────
# Empotramiento
verax_restraints = np.array([
    [0,       1, 1, 1],
])
# Empotramiento
lator_restraints = np.array([
    [0,       1, 1, 1, 1],
])


# ── Cargas nodales ────────────────────────────────────────────────────────
# Carga puntual en la punta sobre la mesa superior
idx = 1
ratios = [0, 1, 2, 4]
r = ratios[idx]
Q = -50e3
nodal_loads = np.array([
    [nelems, 0, 3,    0.0, 0.0,    r*Q, Q, 0.0]
])


# ── Creación y seteo del modelo ─────────────────────────────────────────────
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
stabi = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
mu_cr_ref     = [1.979, 1.742, 1.475, 1.006]
mu_cr_ltbeamn = [1.943, 1.722, 1.469, 1.010]

static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx],
                   "LTbeamN": mu_cr_ltbeamn[idx]})


# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.18, n_sec=2)
plt.show()
