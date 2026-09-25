
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
        # eS(x) cota del centro de corte medidad desde el eje de ref.
        self.eS_i = self.section_i.z_from_ref(self.align, 1)
        self.eS_j = self.section_j.z_from_ref(self.align, 1)
        self.deS  = (self.eS_j - self.eS_i) / self.length

        # derivada de las distancias de los zC de las mesas al centro de corte
        self.daf1 = (self.section_j.af1 - self.section_i.af1) / self.length
        self.daf2 = (self.section_j.af2 - self.section_i.af2) / self.length
        

    def interpolate_at_gauss(self, xi):
        """Interpola sección en punto de Gauss y añade inercias del taper."""
        gsec = interpolate_section(self.section_i, self.section_j, xi)
        # Inercias de taper (Kitipornchair y Trahair 1975)
        I_psi  = 4 * (self.daf1**2 * gsec.Izf1 + self.daf2**2 * gsec.Izf2)
        I_wpsi = 2 * (self.daf1 * gsec.af1 * gsec.Izf1 + self.daf2 * gsec.af2 * gsec.Izf2) 
        I_ypsi = 2 * (self.daf2 * gsec.Izf2 - self.daf1 * gsec.Izf1) # derivado usando la cinematica de beyer2015
        #I_ypsi = 2 * (self.daf1 * gsec.Izf1 - self.daf2 * gsec.Izf2) # original
        
        gsec.update_tapered_inertias(I_psi, I_wpsi, I_ypsi)
        return gsec
        

    def compute_verax_T(self):
        """ 
        Matriz transformacion (6x6) 
        DOFs centroidales --> DOFs eje de ref.
        eC positiva si el eje de referencia está por encima del centroide
        """
        eCi = -self.section_i.z_from_ref(self.align, 0)
        eCj = -self.section_j.z_from_ref(self.align, 0)
        self.T_vrx = np.eye(6)
        self.T_vrx[0, 2] = -eCi # ui_ref = ui_C - eCi * w,x_i
        self.T_vrx[3, 5] = -eCj # uj_ref = uj_C - eCj * w,x_j

    
    def compute_lator_T(self):
        """
        Matriz de transformación (8x8)
        DOFs centro de corte --> DOFs eje de ref.
        eS positivo si el SC esta por encima del centroide
        """
        eSi = self.eS_i
        eSj = self.eS_j
        deS = self.deS

        self.T_ltr = np.eye(8)
        self.T_ltr[0, 2] = -eSi          
        self.T_ltr[1, 2] = -deS          
        self.T_ltr[1, 3] = -eSi          
        self.T_ltr[4, 6] = -eSj
        self.T_ltr[5, 6] = -deS
        self.T_ltr[5, 7] = -eSj
    
    
    def compute_verax_B0(self, xi):
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

    
    def compute_lator_B0(self, xi):
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


    def compute_lator_Bg(self, xi):
        """ Matriz deformacion-desplazamiento geometrica (3x8), ε_g = [v,x  θ,x  θ]"""
        L  = self.length
        N  = N_hermite(xi)
        dN = dN_hermite(xi)

        B = np.zeros((3,8))
        # Pendiente lateral: dv/dx
        B[0, 0::4] = dN[0::2] / L
        B[0, 1::4] = dN[1::2]
        # Torsión: γ = dθ/dx
        B[1, 2::4] = dN[0::2] / L
        B[1, 3::4] = dN[1::2]
        # Giro torsional: θ
        B[2, 2::4] = N[0::2]
        B[2, 3::4] = N[1::2] * L

        return B


    def compute_verax_D0(self, section):
        """ Matriz constitutiva axial-flexión vertical con acoplamiento por excentricidad (2x2)"""
        eC  = section.z_from_ref(self.align, 0) # offset del centroide respecto al eje de referencia
        EA  = self.mater.E * section.A
        EIy = self.mater.E * section.Iy

        return np.array([
            [ EA,          -EA * eC         ],
            [-EA * eC,      EIy + EA * eC**2]
        ])
    
    
    def compute_lator_D0(self, section):
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


    def compute_lator_Dg(self, xi, section):
        """ Matriz de esfuerzos y cargas (3x3) """
        # Esfuerzos internos en los extremos (signo opuesto a la fuerza en el nudo i)
        N1, M1 = -self.forces[0], -self.forces[2]
        N2, M2 =  self.forces[3],  self.forces[5]

        # Axial y momento lineales en el elemento, cortante constante
        N = N1 * (1 - xi) + N2 * xi
        M = M1 * (1 - xi) + M2 * xi
        V = (M1 - M2) / self.length

        # Cargas distribuidas por su altura respecto al centro de corte
        qz_ez = 0.0
        for pos, rez, qzi, qzj in self.qz_loads:
            ez     = section.z_from_ref(1, pos) + rez
            qz_ez += (qzi * (1 - xi) + qzj * xi) * ez

        zS     = section.zS
        i02    = section.i0**2
        beta_z = section.beta_z

        return np.array([
            [N,            N * zS + M,                 -V   ],
            [N * zS + M,   N * i02 - 2 * beta_z * M,    0   ],
            [-V,           0,                          qz_ez]
        ])


    def compute_K0_matrices(self):
        """ Matrices de rigidez axial-flexion vertical (6x6) y torsion-flexion lateral (8x8)"""
        L = self.length

        for xi, w in zip(self.gpoints, self.gweights):
            # Interpolar sección en punto de Gauss
            section = self.interpolate_at_gauss(xi)

            # Matrices constitutivas
            D0_vrx = self.compute_verax_D0(section)
            D0_ltr = self.compute_lator_D0(section)

            # Matrices de deformación-desplazamiento
            B0_vrx = self.compute_verax_B0(xi)
            B0_ltr = self.compute_lator_B0(xi)

            # Acumular contribuciones
            self.K0_vrx += (B0_vrx.T @ D0_vrx @ B0_vrx) * w * L
            self.K0_ltr += (B0_ltr.T @ D0_ltr @ B0_ltr) * w * L
        
        # Traslacion de K0_vrx al centroide y de K0_ltr al eje de referencia
        self.K0_vrx = self.T_vrx.T @ self.K0_vrx @ self.T_vrx
        self.K0_ltr = self.T_ltr.T @ self.K0_ltr @ self.T_ltr


    def update_lator_Kg(self):
        """ Matriz geometrica torsion-flexion lateral (8x8)"""
        L  = self.length
        Kg = np.zeros((8,8))

        for xi, w in zip(self.gpoints, self.gweights):
            # Interpolar sección en punto de Gauss
            section = self.interpolate_at_gauss(xi)

            Dg_ltr = self.compute_lator_Dg(xi, section)
            Bg_ltr = self.compute_lator_Bg(xi)

            Kg += (Bg_ltr.T @ Dg_ltr @ Bg_ltr) * w * L

        # Traslacion de la matriz geometrica al eje de referencia
        self.Kg_ltr = self.T_ltr.T @ Kg @ self.T_ltr


    def add_loads(self, qxpos, qzpos, qxrz, qzrz, qxi, qzi, qxj, qzj):
        """Acumula una carga distribuida en coordenadas locales."""
        self.qz_loads.append((int(qzpos), qzrz, qzi, qzj))

        # excentricidad positiva (+z) y carga axial positiva (traccion) generan momentos negativos
        qxezi = self.section_i.z_from_ref(self.align, int(qxpos)) + qxrz
        qxezj = self.section_j.z_from_ref(self.align, int(qxpos)) + qxrz
        mi = - qxi * qxezi
        mj = - qxj * qxezj

        equi_loads  = self.compute_equivalent_loads(qxi, qzi, qxj, qzj, mi, mj)
        self.loads += self.T_vrx.T @ equi_loads # trasladar cargas al eje centroidal        
