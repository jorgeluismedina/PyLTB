
import numpy as np
import scipy as sp
from pyltb.elements.base_beam import Beam
from pyltb.sections.section_utils import interpolate_section
from pyltb.shape_funcs import N_hermite, dN_hermite, ddN_hermite
from pyltb.gauss_quad import gauss_1d



class BeamNP(Beam):
    def __init__(self, mater, section_i, section_j, coords, conec, 
                 vrx_dofs, ltr_dofs, align=0):
        super().__init__(mater, coords, conec, vrx_dofs, ltr_dofs)
        self.section_i = section_i
        self.section_j = section_j
        self.align     = align

        self.init_geometry()

        self.gpoints, self.gweights = gauss_1d(4)

        # Inicializar matrices de rigidez y geometricas
        self.compute_verax_T()
        self.compute_lator_T()
        self.compute_K0_matrices()     
      

    def init_geometry(self):
        """ Calcula las pendientes de las secciones"""
        # a(x) cota del centro de corte medidad desde el eje de ref.
        self.aS_i = self.section_i.z_from_ref(self.align, 1)
        self.aS_j = self.section_j.z_from_ref(self.align, 1)
        self.daS  = (self.aS_j - self.aS_i) / self.length

        aT_i = abs(self.section_i.zf1 - self.section_i.zS)
        aB_i = abs(self.section_i.zf2 - self.section_i.zS)
        aT_j = abs(self.section_j.zf1 - self.section_j.zS)
        aB_j = abs(self.section_j.zf2 - self.section_j.zS)
        self.daT = (aT_j - aT_i) / self.length
        self.daB = (aB_j - aB_i) / self.length
        #self.dzS = (self.section_j.zS - self.section_i.zS) / self.length
        
    

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
    

    def compute_verax_T(self):
        """ 
        Matriz transformacion (6x6) de DOFs centroidales a DOFs del eje de ref.
        e positiva si el eje de referencia está por encima del centroide
        """
        ei = -self.section_i.z_from_ref(self.align, 0)
        ej = -self.section_j.z_from_ref(self.align, 0)
        self.T_vrx = np.eye(6)
        self.T_vrx[0, 2] = -ei # ui_ref = ui_G - ei * θi
        self.T_vrx[3, 5] = -ej # uj_ref = uj_G - ej * θj
        #return T
    
    def compute_lator_T(self):
        """
        Matriz de transformación (8x8) de DOF centroidales al centrode corte.
        Incluye el efecto de la pendiente dzS.
        zS positivo si el SC esta por encima del centroide
        """
        ai = self.aS_i
        aj = self.aS_j
        da = self.daS
        #zS_i = self.section_i.zS
        #zS_j = self.section_j.zS
        #dzS  = self.dzS
        self.T_ltr = np.eye(8)
        self.T_ltr[0, 2] = -ai#-zS_i          # v_S = v_ref - zS_i * θ
        self.T_ltr[1, 2] = -da#-dzS           # ∂v'_S/∂θ  (por la derivada de zS)
        self.T_ltr[1, 3] = -ai#-zS_i          # ∂v'_S/∂θ'
        self.T_ltr[4, 6] = -aj#-zS_j
        self.T_ltr[5, 6] = -da#-dzS
        self.T_ltr[5, 7] = -aj#-zS_j
    
    
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
            self.K0_vrx += (B_vrx.T @ D_vrx @ B_vrx) * w * L
            self.K0_ltr += (B_ltr.T @ D_ltr @ B_ltr) * w * L
        
        # Trasalacion de las matrices de rigidez al centroide
        self.K0_vrx = self.T_vrx.T @ self.K0_vrx @ self.T_vrx 
        self.K0_ltr = self.T_ltr.T @ self.K0_ltr @ self.T_ltr


    def update_lator_Kg(self):
        """ Matriz geometrica Torsion-Flexion lateral (8x8)"""
        L = self.length
        
        N1 = -self.forces[0] # Axial izquierda
        M1 = -self.forces[2] # Momento izquierd
        N2 =  self.forces[3] # Axial derecha
        M2 =  self.forces[5]  # Momento derecha
        Vz = (M1 - M2) / L  # Cortante

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
            term_V = -Vz * (np.outer(vec_dv, vec_t) + np.outer(vec_t, vec_dv))

            # Aporte de las cargas distribuidas
            term_Q = qzez * qz_xi * np.outer(vec_t, vec_t)   # ec. (20) Beyer — θ²

            self.Kg_ltr += (term_N + term_M + term_V + term_Q) * w * L
            
        # Traslacion de la matriz geometrica lateral-torsional al centroide
        self.Kg_ltr = self.T_ltr.T @ self.Kg_ltr @ self.T_ltr


    def add_loads(self, qxpos, qzpos, qxrz, qzrz, qxi, qzi, qxj, qzj):
        """ Añade cargas en coordenadas locales """
        self.load_ints = np.array([qxi, qzi, qxj, qzj], dtype=float) # intensidades de carga
        self.load_pos  = np.array([qxpos, qzpos], dtype=int)         # posiciones de carga
        self.load_rez  = np.array([qxrz, qzrz], dtype=float)         # excentricidad relativa de carga

        # excentricidad positiva (+z) y carga axial positiva (traccion) generan momentos negativos
        qxezi = self.section_i.z_from_ref(self.align, int(qxpos)) + qxrz
        qxezj = self.section_j.z_from_ref(self.align, int(qxpos)) + qxrz
        mi = - qxi * qxezi
        mj = - qxj * qxezj

        self.compute_equivalent_loads(qxi, qzi, qxj, qzj, mi, mj)
        self.loads = self.T_vrx.T @ self.loads # trasladar cargas al eje centroidal        
