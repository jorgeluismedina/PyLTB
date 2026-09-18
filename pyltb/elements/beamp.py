
import numpy as np
from pyltb.elements.base_beam import Beam
from pyltb.shape_funcs import N_hermite, dN_hermite


class BeamP(Beam):
    def __init__(self, mater, section, coords, conec, vrx_dofs, ltr_dofs, align=0):
        super().__init__(mater, coords, conec, vrx_dofs, ltr_dofs)
        self.section = section
        self.align   = align
        self.set_dof_indices()
        self.compute_lator_T()

        # Inicializar matrices de rigidez y geometricas
        self.compute_verax_K0()
        self.compute_lator_K0()

    
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


    def compute_lator_T(self):
        """
        Matriz de transformación (8x8)
        DOFs eje de referencia --> DOFs centro de corte:
        """
        eS = self.section.z_from_ref(self.align, 1)
        self.T_ltr = np.eye(8)
        self.T_ltr[0, 2] = -eS
        self.T_ltr[1, 3] = -eS
        self.T_ltr[4, 6] = -eS
        self.T_ltr[5, 7] = -eS


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
        self.K0_vrx[self.idx_uu] = EA * axial_base
        self.K0_vrx[self.idx_ww] = EIy * bending_base
    
    
    def compute_lator_K0(self):
        """ Matriz de rigidez flexion lateral (v, v,x) y torsion (theta, theta,x) (8x8) """
        EIz = self.mater.E * self.section.Iz
        GIt = self.mater.G * self.section.It
        EIw = self.mater.E * self.section.Iw

        # Matrices base
        bending_base = self.ddNiddNj_matrix()
        torsion_base = self.dNidNj_matrix()

        # Ensamblaje de matriz de rigidez flexión lateral y torsión con acoplamiento
        self.K0_ltr[self.idx_vv] = EIz * bending_base                      # Bloque v-v (Flexión lateral)
        self.K0_ltr[self.idx_tt] = EIw * bending_base + GIt * torsion_base # Bloque t-t (Torsión = Warping + St.Venant)
        # Traslacion de la matriz al eje de referencia
        self.K0_ltr = self.T_ltr.T @ self.K0_ltr @ self.T_ltr
        
    
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
        NNi = self.NiNj_1_xi_matrix()
        NNj = self.NiNj_xi_matrix()

        KgQ = np.zeros((8, 8))
        for pos, rez, qzi, qzj in self.qz_loads:
            ez     = self.section.z_from_ref(1, pos) + rez   # altura respecto al centro de corte
            Q_base = (qzi*NNi + qzj*NNj)   
            KgQ[self.idx_tt] += Q_base * ez # Bloque t-t (torsion)

        return KgQ


    def update_lator_Kg(self):
        """ Actualiza la matriz geometrica con fuerzas internas """
        KgN  = self.compute_lator_KgN()
        KgMV = self.compute_lator_KgMV()
        KgQ  = self.compute_lator_KgQ()
        Kg_ltr = KgN + KgMV + KgQ
        self.Kg_ltr = self.T_ltr.T @ Kg_ltr @ self.T_ltr
    
    

    def add_loads(self, qxpos, qzpos, qxrz, qzrz, qxi, qzi, qxj, qzj):
        """ Añade cargas en coordenadas locales """
        self.qz_loads.append((int(qzpos), qzrz, qzi, qzj))

        # excentricidad positiva (+z) y carga axial positiva (traccion) generan momentos negativos
        qxez = self.section.z_from_ref(0, int(qxpos)) + qxrz
        mi =  - qxi * qxez
        mj =  - qxj * qxez

        self.loads += self.compute_equivalent_loads(qxi, qzi, qxj, qzj, mi, mj)        
