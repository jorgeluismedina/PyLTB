
import numpy as np
from pyltb.shape_funcs import N_hermite, dN_hermite

class Beam():
    def __init__(self, mater, coords, conec, vrx_dofs, ltr_dofs):
        self.mater  = mater
        self.coords = coords 
        self.conec  = conec
        self.length = np.linalg.norm(coords[1] - coords[0])
        self.vrx_dofs = vrx_dofs # vertical deflection and axial displacement DOFs
        self.ltr_dofs = ltr_dofs # lateral deflection and torsion DOFs

        # inicializacion de matrices
        self.K0_vrx = np.zeros((6, 6)) #None
        self.K0_ltr = np.zeros((8, 8)) #None
        self.Kg_ltr = np.zeros((8, 8)) #None
        
        # Cantidades centroidales del problema estatico
        self.loads  = np.zeros(6)
        self.forces = np.zeros(6)
        self.disps  = np.zeros(6)

        # Parametros de cargas en el plano
        self.load_ints = np.zeros(4, dtype=float) # intensidades de carga
        self.load_pos  = np.zeros(2, dtype=int)   # posiciones de carga
        self.load_rez  = np.zeros(2, dtype=float) # excentricidad relativa de carga


    def compute_equivalent_loads(self, qxi, qzi, qxj, qzj, mi, mj):
        """
        Calcula las fuerzas nodales equivalentes para una carga distribuida
        linealmente variable en el sistema local del elemento.

        Parámetros
        ----------
        qxi, qxj : (float) Intensidad de la carga axial en el nodo i y en el nodo j [N/m].
        qzi, qzj : (float) Intensidad de la carga transversal en el nodo i y en el nodo j [N/m].
        mi, mj   : (float) Momento debido a excentricidad de carga axial en los nodos i y j [N*m].

        """
        L = self.length
        self.loads[0] =  (qxi/3 + qxj/6) * L
        self.loads[1] =  (7*qzi + 3*qzj) * L / 20    -  0.5 * (mi + mj)
        self.loads[2] =  (3*qzi + 2*qzj) * L**2 / 60 + L/12 * (mi - mj)
        self.loads[3] =  (qxj/3 + qxi/6) * L
        self.loads[4] =  (3*qzi + 7*qzj) * L / 20    +  0.5 * (mi + mj)
        self.loads[5] = -(2*qzi + 3*qzj) * L**2 / 60 + L/12 * (mj - mi)

        #self._finalize_loads()


    def calculate_forces(self, glob_disps):
        self.disps = glob_disps
        self.forces = self.K0_vrx @ glob_disps - self.loads
        self.disps[np.abs(self.disps) < 1e-12] = 0
        self.forces[np.abs(self.forces) < 1e-9] = 0


    def get_fields(self):
        L  = self.length
        x  = np.linspace(0,L,2) # 3 puntos nomas
        xi = x/L

        # Obtener funciones de forma y sus derivadas
        Nh  = N_hermite(xi)      # (4, n_points)
        dNh = dN_hermite(xi)    # (4, n_points)

        # Fuerzas internas del elemento
        Ni = -self.forces[0] 
        Vi = -self.forces[1]
        Mi = -self.forces[2]
        Nj =  self.forces[3]
        Vj =  self.forces[4]
        Mj =  self.forces[5]

        # Diagrama de axil (lineal)
        N_diag = (1 - xi) * Ni + xi * Nj

        # Diagrama de momento (interpolacion cubica)
        M_diag = (Nh[0] * Mi +
                  Nh[1] * L * Vi +
                  Nh[2] * Mj +
                  Nh[3] * L * Vj)

        # Diagrama de cortante (derivada del momento)
        V_diag = (dNh[0] * Mi / L +
                  dNh[1] * Vi +
                  dNh[2] * Mj / L +
                  dNh[3] * Vj)

        # Desplazamiento Axial: Interpolacion lineal
        u = (1 - xi) * self.disps[0] + xi * self.disps[3]

        # Desplazamiento Vertical: Interpolacion cubica
        w =  (Nh[0]*self.disps[1] + 
              Nh[1]*L*self.disps[2] + 
              Nh[2]*self.disps[4] + 
              Nh[3]*L*self.disps[5])

        # Tolerancias relativas por tipo de diagrama
        eps = 1e-10
        tol_N = eps * max(abs(Ni), abs(Nj), 1.0)
        tol_V = eps * max(abs(Mi), abs(Mj), 1.0) / L
        tol_M = eps * max(abs(Mi), abs(Mj), 1.0)
        tol_u = eps * max(abs(self.disps[0]), abs(self.disps[3]), 1e-15)
        tol_w = eps * max(abs(self.disps[1]), abs(self.disps[4]), 1e-15)

        N_diag[np.abs(N_diag) < tol_N] = 0.0
        V_diag[np.abs(V_diag) < tol_V] = 0.0
        M_diag[np.abs(M_diag) < tol_M] = 0.0
        u[np.abs(u) < tol_u] = 0.0
        w[np.abs(w) < tol_w] = 0.0

        fields = np.vstack([x + self.coords[0], N_diag, V_diag, M_diag, u, w])

        return fields

