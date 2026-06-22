
import numpy as np

class ISection_MS:
    def __init__(self, h, bf1, bf2, tw, tf1, tf2, r1, r2):
        self.h   = h     # total height
        self.bf1 = bf1   # top flange width
        self.bf2 = bf2   # bottom flange width
        self.tw  = tw    # web thick
        self.tf1 = tf1   # top flange thick
        self.tf2 = tf2   # botttom flange thick
        self.r1  = r1    # top fillets radious
        self.r2  = r2    # bottom fillets radious
        self.compute_basic()
        self.compute_area()
        self.compute_gravity_center()
        self.compute_bending_inertias()
        self.compute_shear_center()
        self.compute_torsional_inertia()
        self.compute_warping_inertia()
        self.compute_polar_radius()
        self.compute_wagner_coeff()

        self.I_psi  = 0.0
        self.I_wpsi = 0.0
        self.I_ypsi = 0.0

    def compute_basic(self):
        # propiedades ala superior
        self.Af1  = self.bf1 * self.tf1
        self.Iyf1 = self.bf1 * self.tf1**3 / 12
        self.Izf1 = self.tf1 * self.bf1**3 / 12
        self.zGf1 = self.h - self.tf1 / 2
        # propiedades ala inferior
        self.Af2  = self.bf2 * self.tf2
        self.Iyf2 = self.bf2 * self.tf2**3 / 12
        self.Izf2 = self.tf2 * self.bf2**3 / 12
        self.zGf2 = self.tf2 / 2
        # propiedades alma
        self.hw = self.h - self.tf1 - self.tf2
        self.Aw = self.hw * self.tw
        self.Iyw = self.tw * self.hw**3 / 12
        self.Izw = self.hw * self.tw**3 / 12
        self.zGw = self.tf2 + self.hw/2
        # terminos repetidos
        one_qpi = 1 - np.pi/4
        four_pi = 4 * one_qpi
        # propiedades fillets superiores
        self.Ar1 = one_qpi * self.r1**2
        self.Ir1 = (1/3 - np.pi/16 - 1/(9*four_pi)) * self.r1**4
        self.vr1 = (1 - 2/(3*four_pi)) * self.r1
        self.zGr1 = self.h - self.tf1 - self.vr1
        # propiedades fillets inferiores
        self.Ar2 = one_qpi * self.r2**2
        self.Ir2 = (1/3 - np.pi/16 - 1/(9*four_pi)) * self.r2**4
        self.vr2 = (1 - 2/(3*four_pi)) * self.r2
        self.zGr2 = self.tf2 + self.vr2

    def compute_area(self):
        self.A = self.Af1 + self.Af2 + self.Aw + 2*self.Ar1 + 2*self.Ar2

    def compute_gravity_center(self):
        self.zG = (self.Af1 * self.zGf1 + 
                   self.Af2 * self.zGf2 +
                   self.Aw * self.zGw +
                   2 * self.Ar1 * self.zGr1 +
                   2 * self.Ar2 * self.zGr2) / self.A
        # centroides de las partes respecto al centroide total
        self.zf1 = self.zGf1 - self.zG
        self.zf2 = self.zGf2 - self.zG
        self.zw = self.zGw - self.zG
        self.zr1 = self.zGr1 - self.zG
        self.zr2 = self.zGr2 - self.zG

    def compute_bending_inertias(self):
        Iyf1 = self.Iyf1 + self.Af1 * self.zf1**2
        Iyf2 = self.Iyf2 + self.Af2 * self.zf2**2
        Iyw  = self.Iyw + self.Aw * self.zw**2
        Iyr1 = self.Ir1 + self.Ar1 * self.zr1**2
        Iyr2 = self.Ir2 + self.Ar2 * self.zr2**2
        Izr1 = self.Ir1 + self.Ar1 * (self.tw/2 + self.vr1)**2
        Izr2 = self.Ir2 + self.Ar2 * (self.tw/2 + self.vr2)**2

        self.Iy = Iyf1 + Iyf2 + Iyw + 2*Iyr1 + 2*Iyr2
        self.Iz = self.Izf1 + self.Izf2 + self.Izw + 2*Izr1 + 2*Izr2

    def compute_shear_center(self): # respecto del centroide
        self.zS = (self.Izf1 * self.zf1 + 
                   self.Izf2 * self.zf2 +
                   self.Izw * self.zw +
                   2 * self.Ir1 * self.zr1 +
                   2 * self.Ir2 * self.zr2) / self.Iz

    def compute_torsional_inertia(self):
        # Saint-Venant base
        self.It = (self.bf1 * self.tf1**3 +
                   self.bf2 * self.tf2**3 +
                   self.hw  * self.tw**3) / 3
        #print(self.It)
        # correction functions
        def alpha(tf, r):
            return (0.2204 * (self.tw / tf) + 
                    0.1355 * (r / tf) -
                    0.0865 * (self.tw * r / tf**2) -
                    0.0725 * (self.tw / tf)**2)
        
        def D(tf, r):
            return ((tf+r)**2 + self.tw*(r+self.tw/4)) / (2*r+tf)
        
        if self.r1 > 0.0:
            alpha1 = alpha(self.tf1, self.r1)
            D1 = D(self.tf1, self.r1)
            self.It += alpha1 * D1**4 - 0.21*self.tf1**4

        if self.r2 > 0.0:
            alpha2 = alpha(self.tf2, self.r2)
            D2 = D(self.tf2, self.r2)
            self.It += alpha2 * D2**4 - 0.21*self.tf2**4
        

    def compute_warping_inertia(self):
        I_ratio = self.Izf1 * self.Izf2 / (self.Izf1 + self.Izf2)
        hs = (self.zGf1 - self.zGf2)
        self.Iw = I_ratio * hs**2

    def compute_polar_radius(self):
        self.i0 = np.sqrt((self.Iy + self.Iz) / self.A + self.zS**2)

    def compute_wagner_coeff(self):
        sf1 = self.zf1 * (self.Izf1 + self.Af1*self.zf1**2 + 3*self.Iyf1)
        sf2 = self.zf2 * (self.Izf2 + self.Af2*self.zf2**2 + 3*self.Iyf2)
        sw = self.zw * (self.Izw + self.Aw*self.zw**2 + 3*self.Iyw)
        sr1 = 2 * self.zr1 * (4*self.Ir1 + self.Ar1*self.zr1**2)
        sr2 = 2 * self.zr2 * (4*self.Ir2 + self.Ar2*self.zr2**2)

        self.beta_z = 1 / (2*self.Iy) * (sf1 + sf2 + sw + sr1 + sr2) - self.zS

    def update_tapered_inertias(self, I_psi, I_wpsi, I_ypsi):
        self.I_psi  = I_psi
        self.I_wpsi = I_wpsi
        self.I_ypsi = I_ypsi

  
    def z_from_ref(self, ref, pos):
        """
        Calcula la altura de un punto `pos` respecto a un eje de referencia `ref`
        pos_code:
            0 → centroide
            1 → centro de corte
            2 → mesa inferior
            3 → mesa superior
        """

        heights = np.array([
            0.0, 
            self.zS,
            -self.zG,
            self.h - self.zG
        ]) # referenciado todo al centroide

        return heights[pos] - heights[ref]



    def summary(self):
        print("\n" + "="*55)
        print(" I-SECTION (MONOSYMMETRIC) – GEOMETRY & PROPERTIES")
        print("="*55)

        # --- Geometry ---
        print("\n[ Geometry ]")
        print(f"  h   = {self.h:.4f}")
        print(f"  bf1 = {self.bf1:.4f}")
        print(f"  bf2 = {self.bf2:.4f}")
        print(f"  tf1 = {self.tf1:.4f}")
        print(f"  tf2 = {self.tf2:.4f}")
        print(f"  tw  = {self.tw:.4f}")
        print(f"  hw  = {self.hw:.4f}")
        print(f"  r1  = {self.r1:.4f}")
        print(f"  r2  = {self.r2:.4f}")

        # --- Properties ---
        print("\n[ Properties ]")
        print(f"  A  = {self.A:.4e}")
        print(f"  Iy = {self.Iy:.4e}")
        print(f"  Iz = {self.Iz:.4e}")
        print(f"  It = {self.It:.4e}")
        print(f"  Iw = {self.Iw:.4e}")
        print(f"  βz = {self.beta_z:.6f}")
        #print()
        print(f"  zG = {self.zG:.6f}  (from bottom fiber)")
        print(f"  zS = {self.zS:.6f}  (relative to centroid)")
        print(f"  i0 = {self.i0:.6f}  (respect to shear center)")
        
        # --- Tapered effect Properties ---
        print("\n[ Tapered Effects Properties ]")
        print(f"  I_psi  = {self.I_psi:.4e}")
        print(f"  I_wpsi = {self.I_wpsi:.4e}")
        print(f"  I_ypsi = {self.I_ypsi:.4e}")

        print("\n" + "="*55 + "\n")