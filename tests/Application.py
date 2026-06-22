

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


# ── Materiales ────────────────────────────────────────────────────────────
materials = [Material(E=2.1e11, nu=0.3, dens=1.0)]
 
# ── Secciones ─────────────────────────────────────────────────────────────
sec_A = ISection_MS(h=0.324, bf1=0.27, bf2=0.27, tw=0.006, tf1=0.012, tf2=0.012, r1=0, r2=0)
sec_B = ISection_MS(h=0.924, bf1=0.27, bf2=0.27, tw=0.006, tf1=0.012, tf2=0.012, r1=0, r2=0)

# ── Malla ─────────────────────────────────────────────────────────────────
L, nelems = 9, 24
nodes    = np.linspace(0, L, nelems + 1)
sections = interpolate_multiple_sections(sec_A, sec_B, nodes / L)


# ── Modelo ────────────────────────────────────────────────────────────────
kv = materials[0].E * sec_B.Iy * 1e6

elements_data = np.array([[1, 0, e, e+1] for e in range(nelems)])

vrx_restraints = np.array([[0,        1, 1, 0],
                           [nelems,   0, 1, 0]])

ltr_restraints = np.array([[0,            1, 0, 1, 0],
                           [2*nelems//3,  1, 0, 1, 0],
                           [nelems,       1, 0, 1, 0]])

springs_data = np.array([[nelems//3,  3,  kv, 0.0, 0.0, 0.0]])

nodal_loads = np.array([[nelems,  0, 0,  0.0, 0.0,  -210.6e3, 0.0, -683.2e3]])
elem_loads = np.array([[e, 0, 3,  0.0, 0.0,  0.0, 6e3, 0.0, 6e3] for e in range(nelems)])
 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(nodes)
model.add_tapered_elements(elements_data, align=3)
model.add_verax_restraints(vrx_restraints)
model.add_lator_restraints(ltr_restraints)
model.add_lateral_springs(springs_data)
model.add_nodal_loads(nodal_loads)
model.add_elem_loads(elem_loads)
model.summary()

# ── Resolver ──────────────────────────────────────────────────────────────
static = StaticSolver(model).solve()
stabi  = StabilitySolver(model).solve()

# ── Resultados ────────────────────────────────────────────────────────────
static.summary()
stabi.summary(ref={"LTBeamN": 3.009})

# ── Plots ─────────────────────────────────────────────────────────────────
static.plot()
stabi.plot(imode=0, scale=0.13)
plt.show()