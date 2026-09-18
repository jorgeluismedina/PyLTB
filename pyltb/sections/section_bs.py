
import numpy as np


# Seccion I bi-simetrica
class ISection_BS:
    def __init__(self, h, bf, tw, tf, r):
        self.h  = h     # total height
        self.bf = bf    # flanges width
        self.tw = tw    # web thick
        self.tf = tf    # flanges thick
        self.r  = r     # radius of fillets

        self.hw = h - 2*tf # altura del alma entre flanges
        self.zG = h / 2
        self.zS = 0.0
        self.beta_z = 0.0

        self.It_type = "plates"   # "villette" | "darwish" | "plates"
        self.compute_area()
        self.compute_bending_inertias()
        self.compute_torsional_inertia()
        self.compute_warping_inertia()
        self.compute_polar_radius()


    def compute_area(self):
        A_web = self.tw * self.hw
        A_flanges = 2 * self.bf * self.tf
        A_corners = self.r**2 * (4 - np.pi) # area de los fillets
        self.A = A_web + A_flanges + A_corners
    
    def compute_bending_inertias(self):
        four_pi = 4 - np.pi
        term = 4*self.r**4 * (1/3 - np.pi/16 - 1/(9*four_pi))

        term1y = 1/12 * (self.bf * self.h**3 - (self.bf - self.tw) * self.hw**3)
        term2y = four_pi*self.r**2 * (self.zG - self.tf - self.r + 2*self.r/(3*four_pi))**2
        self.Iy = term1y + term + term2y

        term1z = 1/6*self.tf*self.bf**3 + 1/12*self.tw**3*self.hw
        term2z = four_pi*self.r**2 * (self.tw/2 + self.r - 2*self.r/(3*four_pi))**2
        self.Iz = term1z + term + term2z


    def It_villette(self):
        """ Villette (2011) - formula que usa LTBeamN """
        def ItV(d1, d2):
            dmax, dmin = max(d1, d2), min(d1, d2)
            return dmax * dmin**3 / 3 * (1 - dmin/dmax * (0.633 - 0.055 * dmin**3 / dmax**3))

        ratio  = (6*self.tw + self.tf) / self.bf
        alphaV = 4.0 if ratio <= 1.0 else 8 / (1 + ratio)

        # rectangulos: alas + alma completa - solape alma/alas
        It = (2 * ItV(self.bf, self.tf) +
              ItV(self.h, self.tw) -
              2 * (self.tw / self.bf)**2 * ItV(self.tw, self.tf))

        # nudos alma-ala con fillets (se anulan si r = 0)
        It += 2 * alphaV * (ItV(self.tw + 0.4*self.r, self.tf + 0.4*self.r) - ItV(self.tw, self.tf))
        return It

    def It_darwish(self):
        """ Darwish & Johnston (1965) """
        alpha = (-0.042 +
                 0.2204 * (self.tw / self.tf) +
                 0.1355 * (self.r / self.tf) -
                 0.0865 * (self.tw * self.r / self.tf**2) -
                 0.0725 * (self.tw / self.tf)**2)

        D = ((self.tf + self.r)**2 + self.tw * (self.r + self.tw/4)) / (2*self.r + self.tf)

        # Saint-Venant base + nudos alma-ala
        It = (2 * self.bf * self.tf**3 + self.hw * self.tw**3) / 3
        It += 2 * (alpha * D**4 - 0.21 * self.tf**4)
        return It

    def It_plates(self):
        """ Suma simple de placas: Saint-Venant sin correccion de nudos """
        return (2 * self.bf * self.tf**3 + self.hw * self.tw**3) / 3

    def compute_torsional_inertia(self):
            """ Inercia de torsion de Saint-Venant """
    
            formulas = {"villette": self.It_villette,
                        "darwish":  self.It_darwish,
                        "plates":   self.It_plates}
            self.It = formulas[self.It_type]()
        
    def compute_warping_inertia(self):
        self.Iw = 0.25 * self.Iz * (self.h - self.tf)**2

    def compute_polar_radius(self): #respecto al centro de corte
        self.i0 = np.sqrt((self.Iy + self.Iz) / self.A)
    
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
        print("\n" + "="*50)
        print(" I-SECTION (BISYMMETRIC) – GEOMETRY & PROPERTIES")
        print("="*50)

        # --- Geometry ---
        print("\n[ Geometry ]")
        print(f"  h  = {self.h:.4f}")
        print(f"  bf = {self.bf:.4f}")
        print(f"  tf = {self.tf:.4f}")
        print(f"  tw = {self.tw:.4f}")
        print(f"  hw = {self.hw:.4f}")
        print(f"  r  = {self.r:.4f}")

        # --- Properties ---
        print("\n[ Properties ]")
        print(f"  A  = {self.A:.4e}")
        print(f"  Iy = {self.Iy:.4e}")
        print(f"  Iz = {self.Iz:.4e}")
        print(f"  It = {self.It:.4e}")
        print(f"  Iw = {self.Iw:.4e}")
        print(f"  βz = {self.beta_z:.6f}")
        print(f"  zG = {self.zG:.6f}  (from bottom fiber)")
        print(f"  zS = {self.zS:.6f}  (relative to centroid)")
        print(f"  i0 = {self.i0:.6f}  (respect to shear center)")
        print("\n" + "="*50 + "\n")
