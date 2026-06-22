
import numpy as np
import scipy as sp
from pyltb.elements.base_beam import Beam
from pyltb.shape_funcs import N_hermite, dN_hermite


class BeamP(Beam):
    def __init__(self, mater, section, coords, conec, vrx_dofs, ltr_dofs):
        super().__init__(mater, coords, conec, vrx_dofs, ltr_dofs)

        self.section = section
        self.align   = 0

        self.init_geometry()
        self.set_dof_indices()

        # Inicializar matrices de rigidez y geometricas
        self.K0_vrx = self.compute_verax_K0()
        self.K0_ltr = self.compute_lator_K0()
        self.Kg_ltr = np.zeros((8, 8))

        # Solo problema estatico
        self.loads  = np.zeros(6)
        self.forces = np.zeros(6)
        self.disps  = np.zeros(6)

        self.load_ints = np.zeros(4, dtype=int)
        self.load_pos  = np.zeros(2, dtype=int)
        self.load_rez  = np.zeros(2, dtype=int)


        
  
    def init_geometry(self):
        vector = self.coords[1] - self.coords[0]
        self.length = sp.linalg.norm(vector)

    
    def set_dof_indices(self):
        # Indices para u (0, 3) y w (1, 2, 4, 5)
        idx_u = [0, 3]
        idx_w = [1, 2, 4, 5]

        # Indices para v (0, 1, 4, 5) y theta (2, 3, 6, 7)
        idx_v = [0, 1, 4, 5]
        idx_t = [2, 3, 6, 7]

        # Indices de submatrices para rigidez y acoplamiento
        self.idx_uu = np.ix_(idx_u, idx_u)
        self.idx_ww = np.ix_(idx_w, idx_w)  
        self.idx_vv = np.ix_(idx_v, idx_v)
        self.idx_tt = np.ix_(idx_t, idx_t)
        self.idx_vt = np.ix_(idx_v, idx_t)
        self.idx_tv = np.ix_(idx_t, idx_v)


    def ddNiddNj_matrix(self):
        """ Integral (Ni'' * Nj'') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 12,   6*L,   -12,   6*L],
            [ 6*L,  4*L**2, -6*L, 2*L**2],
            [-12,  -6*L,    12,   -6*L],
            [ 6*L,  2*L**2, -6*L, 4*L**2]
        ]) / L**3

        return matrix
    
    def dNidNj_matrix(self):
        """ Integral (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 36,   3*L,   -36,   3*L],
            [ 3*L,  4*L**2, -3*L, -L**2],
            [-36,  -3*L,    36,  -3*L],
            [ 3*L, -L**2,  -3*L,  4*L**2]
        ]) / (30*L)

        return matrix
    
    def dNidNj_1_xi_matrix(self):
        """ Integral (1-xi) * (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 18,    0,      -18,    3*L ],
            [ 0,    3*L**2,   0,   -0.5*L**2],
            [-18,    0,       18,   -3*L ],
            [ 3*L, -0.5*L**2,  -3*L,    L**2]
        ]) / (30*L)

        return matrix
    
    def dNidNj_xi_matrix(self):
        """ Integral xi * (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 18,    3*L,    -18,   0   ],
            [ 3*L,   L**2,   -3*L,   -0.5*L**2],
            [-18,   -3*L,     18,   0   ],
            [ 0,    -0.5*L**2,   0,  3*L**2]
        ]) / (30*L)

        return matrix
    
    def dNiNj_matrix(self):
        """ Integral (Ni' * Nj) función de forma cúbica Hermite, asimetrica """
        L = self.length
        matrix = np.array([
            [ -30,    -6*L,     -30,    6*L],
            [  6*L,    0,       -6*L,   L**2],
            [  30,     6*L,      30,   -6*L],
            [ -6*L,   -L**2,     6*L,     0]
        ]) / (60)

        return matrix
    
    def NiNj_1_xi_matrix(self):
        """ Integral (1-xi) * (Ni * Nj) función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 240,      30*L,      54,    -14*L   ],
            [ 30*L,     5*L**2,    12*L,   -3*L**2 ],
            [ 54,       12*L,      72,    -14*L   ],
            [ -14*L,   -3*L**2,   -14*L,    3*L**2  ]
        ]) * L / 840.0
        return matrix
    
    def NiNj_xi_matrix(self):
        """ Integral (xi) * (Ni * Nj) función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 72,      14*L,      54,     -12*L   ],
            [ 14*L,     3*L**2,    14*L,   -3*L**2 ],
            [ 54,       14*L,      240,   -30*L   ],
            [ -12*L,   -3*L**2,   -30*L,   5*L**2  ]
        ]) * L / 840.0
        return matrix


    
    def compute_verax_K0(self):
        """ Matriz de rigidez flexion vertical (w, w,x) y desp. axial (u) (6x6) """
        EA  = self.mater.E * self.section.A
        EIy = self.mater.E * self.section.Iy

        # Matrices base
        axial_base = np.array([[1, -1],[-1, 1]]) / self.length
        bending_base = self.ddNiddNj_matrix()

        # Ensamblaje de matriz de rigidez viga convencional
        K0_vrx = np.zeros((6, 6))
        K0_vrx[self.idx_uu] = EA * axial_base
        K0_vrx[self.idx_ww] = EIy * bending_base

        return K0_vrx
    
    
    def compute_lator_K0(self):
        """ Matriz de rigidez flexion lateral (v, v,x) y torsion (theta, theta,x) (8x8) """
        EIz = self.mater.E * self.section.Iz
        GIt = self.mater.G * self.section.It
        EIw = self.mater.E * self.section.Iw

        # Matrices base
        bending_base = self.ddNiddNj_matrix()
        torsion_base = self.dNidNj_matrix()

        # Ensamblaje de matriz de rigidez flexión lateral y torsión con acoplamiento
        K0_ltr = np.zeros((8, 8))
        K0_ltr[self.idx_vv] = EIz * bending_base                      # Bloque v-v (Flexión lateral)
        K0_ltr[self.idx_tt] = EIw * bending_base + GIt * torsion_base # Bloque t-t (Torsión = Warping + St.Venant)
        
        return K0_ltr
        
    
    def compute_lator_KgN(self): 
        """ Matriz geometrica por carga axial (8x8) """
        zs = self.section.zS
        i02 = self.section.i0**2

        N1 = -self.forces[0]
        N2 =  self.forces[3]
            
        # Matriz base
        N_base = (N1 * self.dNidNj_1_xi_matrix() + 
                  N2 * self.dNidNj_xi_matrix())

        # Ensamblaje de matriz geometrica por carga axial
        KgN = np.zeros((8, 8))
        
        # Bloques diagonales
        KgN[self.idx_vv] = N_base        # Bloque v-v (Flexión lateral)
        KgN[self.idx_tt] = i02 * N_base  # Bloque t-t (Torsión)

        # Bloques de acoplamiento (zc)
        block_vt = zs * N_base
        KgN[self.idx_vt] += block_vt # Bloque vt (Acoplamiento)
        KgN[self.idx_tv] += block_vt # Bloque tv = vt (Acoplamiento)

        return KgN
    

    def compute_lator_KgMV(self):
        """ Matriz geometrica por momento y cortante (KgM + KgV) (8x8) """
        beta_z = self.section.beta_z
        L = self.length

        M1 = -self.forces[2] # Momento en nodo i (signo invertido por carga nodal)
        M2 =  self.forces[5] # Momento en nodo j
        Vz = (M1 - M2) / L
  
        My_base = (M1 * self.dNidNj_1_xi_matrix() + 
                   M2 * self.dNidNj_xi_matrix()) # Integral de My * Ni' * Nj'
        
        Vz_base = Vz * self.dNiNj_matrix() # Integral de Vz * Ni' * Nj (asimetrica)

        # Ensamblaje de matriz geometrica por momento y cortante
        KgMV = np.zeros((8, 8))
        
        # Bloques directos diagonal (t-t), (v-v) = 0
        # Termino: 2 * beta_z * My * t' * t'
        block_tt = - 2 * beta_z * My_base
        KgMV[self.idx_tt] += block_tt

        # Bloques de acoplamiento (v-t y t-v) 
        # Termino: My * v' * t' - Vz * v' * t
        block_vt = My_base - Vz_base 
        KgMV[self.idx_vt] += block_vt
        KgMV[self.idx_tv] += block_vt.T 

        return KgMV
    
    def compute_lator_KgQ(self):
        """ Matriz geométrica por altura de carga transversal distribuida (8x8) """
        qzi  = self.load_ints[1]
        qzj  = self.load_ints[3]

        # Excentricidad de la carga vertical distribuida respecto al eje de referencia
        pos  = self.load_pos[1]
        rez  = self.load_rez[1]
        qzez = self.section.z_from_ref(1, pos) + rez
         
        # qz(xi) = qzi*(1-xi) + qzj*xi
        Q_base = (qzi * self.NiNj_1_xi_matrix() + 
                  qzj * self.NiNj_xi_matrix())

        KgQ = np.zeros((8, 8))       
        KgQ[self.idx_tt] += Q_base * qzez # Bloque t-t (torsion)

        return KgQ


    def update_lator_Kg(self):
        """ Actualiza la matriz geometrica con fuerzas internas """
        KgN  = self.compute_lator_KgN()
        KgMV = self.compute_lator_KgMV()
        KgQ  = self.compute_lator_KgQ()
        self.Kg_ltr = KgN + KgMV + KgQ
    
    

    def add_loads(self, qxpos, qzpos, qxrz, qzrz, qxi, qzi, qxj, qzj):
        """ Añade cargas en coordenadas locales """
        # qxi = intensidad en el nodo i en direccion de la barra
        # qxj = intensidad en el nodo j en direccion de la barra
        # qzi = intensidad en el nodo i en direccion perpendicular de la barra
        # qzj = intensidad en el nodo j en direccion perpendicular de la barra
        # qxpos = posicion (altura) de aplicacion de la carga axial
        # qzpos = posicion (altura) de aplicacion de la carga vertical

        self.load_ints = np.array([qxi, qzi, qxj, qzj], dtype=float)
        self.load_pos  = np.array([qxpos, qzpos], dtype=int)
        self.load_rez  = np.array([qxrz, qzrz], dtype=float)
        
        L = self.length

        self.loads[0] =  (qxi/3 + qxj/6) * L
        self.loads[1] =  (7*qzi + 3*qzj) * L / 20
        self.loads[2] =  (3*qzi + 2*qzj) * L**2 / 60
        self.loads[3] =  (qxj/3 + qxi/6) * L
        self.loads[4] =  (3*qzi + 7*qzj) * L / 20
        self.loads[5] = -(2*qzi + 3*qzj) * L**2 / 60

        # Corrección por excentricidad de carga axial distribuida
        qxez = self.section.z_from_ref(0, int(qxpos)) + qxrz
        # excentricidad positiva (+z) y carga axial positiva (traccion) generan momentos negativos
        mi =  - qxi * qxez
        mj =  - qxj * qxez

        self.loads[1] += -0.5  * (mi + mj)        
        self.loads[2] +=  L/12 * (mi - mj)        
        self.loads[4] +=  0.5  * (mi + mj)        
        self.loads[5] +=  L/12 * (mj - mi)


    def calculate_forces(self, glob_disps):
        # A Coordenadas locales=globales
        self.disps = glob_disps # ya son locales
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
        # M(xi) = N1*Mi + N2*(L*Vi) + N3*Mj + N4*(L*Vj)
        M_diag = (Nh[0] * Mi +
                  Nh[1] * L * Vi +
                  Nh[2] * Mj +
                  Nh[3] * L * Vj)
        
        # Diagrama de cortante (derivada del momento)
        # V = dM/dx = (1/L) * dM/dxi
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
        tol_V = eps * max(abs(Mi), abs(Mj), 1.0) / L   # V ~ M/L
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

    