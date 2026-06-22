
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pyltb.model import StabilityModel
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.solvers.static import StaticSolver
from pyltb.solvers.stability import StabilitySolver


# MATERIALES
materials = [Material(E=2.1e11, nu=0.3, dens=1.0)]

# SECCIONES
section1 = ISection_MS(h=0.61, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]
section2 = ISection_MS(h=0.305, bf1=0.10, bf2=0.18, tw=0.008, tf1=0.010, tf2=0.010, r1=0.00, r2=0.00) #[m]


# MALLA
idx = 0 # índice de longitud a analizar
Ls  = np.array([2, 4, 6, 8, 10])
L   = Ls[idx]
nelems = int(4*L)

nodes = np.linspace(0, L, nelems + 1)
sections = interpolate_multiple_sections(section1, section2, nodes / L)


# MODELO
elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])

verax_restraints = np.array([[0,  1, 1, 1],])
lator_restraints = np.array([[0,  1, 1, 1, 1]])


# Carga vertical Q en el extremo libre sobre el ala superior
nodal_loads = np.array([
    [nelems,  3, 3,   0.0, 0.0,    0.0, -1000.0, 0.0]
])

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
mu_cr_ref     = [77.21, 26.76, 15.08, 9.680, 6.610]
mu_cr_ltbeamn = [76.24, 26.78, 15.03, 9.623, 6.563] # del articulo de Beyer, 2015
mu_cr_ltbeamn = [75.54, 26.55, 14.93, 9.572, 6.537] # con el programa
static.summary()
stabi.summary(ref={"Ref.": mu_cr_ref[idx], 
                   "LTbeamN": mu_cr_ltbeamn[idx]})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.015)
plt.show()


