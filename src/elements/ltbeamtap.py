
import numpy as np
import scipy as sp
from src.elements.base_beam import Beam
from src.sections.section_utils import interpolate_section
from src.shape_funcs import N_hermite, dN_hermite, ddN_hermite
from src.gauss_quad import gauss_1d



class LTBeamTap(Beam):
    def __init__(self, mater, section_i, section_j, coords, conec, 
                 vrx_dofs, ltr_dofs, align=0):
        super().__init__(mater, coords, conec, vrx_dofs, ltr_dofs)

        self.section_i = section_i
        self.section_j = section_j
        self.align     = align

        self.init_geometry()

        self.gpoints, self.gweights = gauss_1d(4)

        # Inicializar matrices de rigidez y geometricas
        self.K0_vrx, self.K0_ltr = self.compute_K0_matrices()
        self.Kg_ltr = np.zeros((8, 8))

        # Solo problema estatico
        self.loads   = np.zeros(6)
        self.forces  = np.zeros(6)
        self.forcesG = np.zeros(6)
        self.disps   = np.zeros(6)
        
        self.load_ints = np.zeros(4, dtype=float)
        self.load_pos  = np.zeros(2, dtype=int)
        self.load_rez  = np.zeros(2, dtype=float)

        

    def init_geometry(self):
        vector = self.coords[1] - self.coords[0]
        self.length = sp.linalg.norm(vector)
      
        aT_i = abs(self.section_i.zf1 - self.section_i.zS)
        aB_i = abs(self.section_i.zf2 - self.section_i.zS)
        aT_j = abs(self.section_j.zf1 - self.section_j.zS)
        aB_j = abs(self.section_j.zf2 - self.section_j.zS)
        self.daT = (aT_j - aT_i) / self.length
        self.daB = (aB_j - aB_i) / self.length
        
    

    def interpolate_at_gauss(self, xi):
        """Interpola sección en punto de Gauss y añade inercias del taper."""
        L     = self.length
        dx    = 1e-2        # 10 mm — paso para diferenciacion numerica
        delta = dx / L
        
        gsec      = interpolate_section(self.section_i, self.section_j, xi)
        sec_plus  = interpolate_section(self.section_i, self.section_j, xi + delta)
        sec_minus = interpolate_section(self.section_i, self.section_j, xi - delta)
        
        # Inercias de taper (Andrade 2005 / Beyer 2015 Apendice A)
        I_psi  = 2 * (sec_plus.Iw - 2*gsec.Iw + sec_minus.Iw) / (delta * L)**2
        I_wpsi = (sec_plus.Iw - sec_minus.Iw) / (2 * delta * L)
        I_ypsi = 2 * (self.daT * gsec.Izf1 - self.daB * gsec.Izf2) # Aproximacion
        
        gsec.update_tapered_inertias(I_psi, I_wpsi, I_ypsi)
        return gsec
        

    def compute_interpolation_vectors(self, xi):
        """ Vectores para ensamblar term-wise la parte de Kg_ltr"""
        L  = self.length
        N  = N_hermite(xi)
        dN = dN_hermite(xi)

        # Vector v' (derivada de la flexión lateral)
        vec_dv = np.zeros(8)
        vec_dv[0::4] = dN[0::2] / L  
        vec_dv[1::4] = dN[1::2]

        # Vector theta' (derivada del giro torsional)
        vec_dt = np.zeros(8)
        vec_dt[2::4] = dN[0::2] / L  
        vec_dt[3::4] = dN[1::2]     
            
        # Vector theta (giro torsional)
        vec_t = np.zeros(8)
        vec_t[2::4] = N[0::2]        
        vec_t[3::4] = N[1::2] * L          

        return vec_dv, vec_t, vec_dt
    
    
    def compute_verax_B(self, xi):
        """ Matriz deformacion-desplazamiento axial flexion vertical (2x6)"""
        L   = self.length
        ddN = ddN_hermite(xi)
        
        B = np.zeros((2,6))
        # Deformación axial: ε = du/dx
        B[0, 0] = -1/L; 
        B[0, 3] =  1/L  
        # Curvatura: κ = d²w/dx²
        B[1, 1] = ddN[0] / L**2; 
        B[1, 2] = ddN[1] / L; 
        B[1, 4] = ddN[2] / L**2; 
        B[1, 5] = ddN[3] / L 

        return B

    
    def compute_lator_B(self, xi):
        """ Matriz deformacion-desplazamiento torsion flexion lateral (3x8)"""
        L   = self.length
        dN  = dN_hermite(xi)
        ddN = ddN_hermite(xi)

        B = np.zeros((3,8))
        # Curvatura lateral: κ_v = d²v/dx²
        B[0, 0::4] = ddN[0::2] / L**2
        B[0, 1::4] = ddN[1::2] / L
        # Curvatura de warping: κ_w = d²θ/dx²
        B[1, 2::4] = ddN[0::2] / L**2
        B[1, 3::4] = ddN[1::2] / L
        # Torsión: γ = dθ/dx
        B[2, 2::4] =  dN[0::2] / L
        B[2, 3::4] =  dN[1::2]

        return B 
    
    def compute_verax_D(self, section):
        """ Matriz constitutiva axial-flexión vertical con acoplamiento por excentricidad (2x2)"""
        e   = section.z_from_ref(self.align, 0) # offset del centroide respecto al eje de referencia
        EA  = self.mater.E * section.A
        EIy = self.mater.E * section.Iy

        return np.array([
            [ EA,          -EA * e        ],
            [-EA * e,       EIy + EA * e**2]
        ])
    
    
    def compute_lator_D(self, section):
        """ Matriz constitutiva torsion flexion lateral (3x3)"""
        EIz = self.mater.E * section.Iz
        EIw = self.mater.E * section.Iw
        GIt = self.mater.G * section.It

        EI_psi  = self.mater.E * section.I_psi
        EI_wpsi = self.mater.E * section.I_wpsi
        EI_ypsi = self.mater.E * section.I_ypsi

        return np.array([
            [EIz,      0,        EI_ypsi],
            [0,        EIw,      EI_wpsi],
            [EI_ypsi,  EI_wpsi,  GIt + EI_psi]
        ])
    


    def compute_K0_matrices(self):
        """ Matriz de rigidez Axial-Flexion vertical (6x6)"""
        """ Matriz de rigidez Torsion-Flexion lateral (8x8)"""
        K0_vrx = np.zeros((6, 6))
        K0_ltr = np.zeros((8, 8))
        L = self.length

        for xi, w in zip(self.gpoints, self.gweights):
            # Interpolar sección en punto de Gauss
            section = self.interpolate_at_gauss(xi)

            # Matrices constitutivas
            D_vrx = self.compute_verax_D(section)
            D_ltr = self.compute_lator_D(section)

            # Matrices de deformación-desplazamiento
            B_vrx = self.compute_verax_B(xi)
            B_ltr = self.compute_lator_B(xi)

            # Acumular contribuciones
            K0_vrx += (B_vrx.T @ D_vrx @ B_vrx) * w * L
            K0_ltr += (B_ltr.T @ D_ltr @ B_ltr) * w * L
        
        return K0_vrx, K0_ltr


    

    def update_lator_Kg(self):
        """ Matriz geometrica Torsion-Flexion lateral (8x8)"""
        Kg_ltr = np.zeros((8, 8))
        L = self.length
        
        N1 = -self.forcesG[0] # Axial izquierda
        M1 = -self.forcesG[2] # Momento izquierd
        N2 =  self.forcesG[3] # Axial derecha
        M2 =  self.forcesG[5]  # Momento derecha
        V_z = (M1 - M2) / L  # Cortante

        qzi = self.load_ints[1]
        qzj = self.load_ints[3]

        for xi, w in zip(self.gpoints, self.gweights):  
            # Interpolar fuerzas internas e intensidad de carga
            M_xi  = M1 * (1 - xi) + M2 * xi
            N_xi  = N1 * (1 - xi) + N2 * xi 
            qz_xi = qzi * (1 - xi) + qzj * xi

            # Propiedades geométricas en la rebanada actual
            section = self.interpolate_at_gauss(xi)
            zS      = section.zS
            i02     = section.i0**2
            beta_z  = section.beta_z

            # Excentricidad de la carga vertical distribuida respecto al eje de referencia
            pos  = self.load_pos[1]
            rez  = self.load_rez[1]
            qzez = section.z_from_ref(1, pos) + rez

            # Vectores de interpolación para ensamblar término a término
            vec_dv, vec_t, vec_dt = self.compute_interpolation_vectors(xi)

            # Ensamblaje numérico de la Ecuación 17 (Beyer et al.)
            # Términos de Fuerza Axial N
            term_N = N_xi * (
                np.outer(vec_dv, vec_dv) + 
                i02 * np.outer(vec_dt, vec_dt) + 
                zS * (np.outer(vec_dv, vec_dt) + np.outer(vec_dt, vec_dv))
            )
            
            # Términos de Momento My
            term_M = M_xi * (
                np.outer(vec_dv, vec_dt) + np.outer(vec_dt, vec_dv) - 
                2 * beta_z * np.outer(vec_dt, vec_dt)
            )
            
            # Término de Cortante Vz
            term_V = -V_z * (np.outer(vec_dv, vec_t) + np.outer(vec_t, vec_dv))

            # Aporte de las cargas distribuidas
            term_Q = qzez * qz_xi * np.outer(vec_t, vec_t)   # ec. (20) Beyer — θ²

            
            Kg_ltr += (term_N + term_M + term_V + term_Q) * w * L
            
        self.Kg_ltr = Kg_ltr

    

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
        qxezi = self.section_i.z_from_ref(self.align, int(qxpos)) + qxrz
        qxezj = self.section_j.z_from_ref(self.align, int(qxpos)) + qxrz
        # excentricidad positiva (+z) y carga axial positiva (traccion) generan momentos negativos
        mi = - qxi * qxezi
        mj = - qxj * qxezj
 
        self.loads[1] += -0.5  * (mi + mj)        
        self.loads[2] +=  L/12 * (mi - mj)        
        self.loads[4] +=  0.5  * (mi + mj)        
        self.loads[5] +=  L/12 * (mj - mi)        


    def calculate_forces(self, glob_disps):
        """ Calcula fuerzas internas en el eje de referencia y respecto al centroide """
        self.disps = glob_disps # ya son locales
        self.forces = self.K0_vrx @ glob_disps - self.loads

        # Momentos corregidos respecto al centroide G
        ei = self.section_i.z_from_ref(self.align, 0)
        ej = self.section_j.z_from_ref(self.align, 0)
        self.forcesG = self.forces.copy()
        self.forcesG[2] += self.forces[0] * ei
        self.forcesG[5] += self.forces[3] * ej
 
        self.disps[ np.abs(self.disps)  < 1e-12] = 0
        self.forces[np.abs(self.forces) < 1e-9 ] = 0
        self.forcesG[np.abs(self.forcesG) < 1e-9 ] = 0


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

        return x, N_diag, V_diag, M_diag, u, w
