
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from pyltb.plotting import plot_diagrams



class StaticSolver():
    def __init__(self, model):
        self.model = model


    def assemble_verax_K0(self):
        """Ensambla matriz de rigidez global."""
        nvrx_dofs = self.model.nvrx_dofs
        K0_vrx = np.zeros((nvrx_dofs, nvrx_dofs))
        for elem in self.model.elements:
            K0_vrx[np.ix_(elem.vrx_dofs, elem.vrx_dofs)] += elem.K0_vrx
        
        return K0_vrx
    

    def assemble_verax_F(self):
        """Ensambla vector de cargas."""
        F = np.zeros(self.model.nvrx_dofs)
        
        if self.model.loaded_nodes:
            dofs, vals = self.model.assemble_global_vec(
                self.model.avrx_dofs, 
                self.model.loaded_nodes, 
                self.model.nodal_loads
            )
            F[dofs] += vals

            for i, node in enumerate(self.model.loaded_nodes):
                dof_Mx = self.model.avrx_dofs[node, 2]
                Fx  = self.model.nodal_loads[i, 0]
                
                if Fx == 0.0:
                    continue
    
                pos    = self.model.nloads_pos[i, 0]
                rez    = self.model.nloads_rez[i, 0]
                align  = self.model.node_align[node]
                sec    = self.model.sections[node]
                fxez   = sec.z_from_ref(0, pos) + rez
                  
                F[dof_Mx] -= Fx * fxez #el trabajo de la fuerza axial tiene que ser negativo

        if self.model.loaded_elems:
            for id_elem in self.model.loaded_elems:
                elem = self.model.elements[id_elem]
                F[elem.vrx_dofs] += elem.loads

        return F

    
    def process_verax_restraints(self):
        """Separa DOFs fijos y libres."""
        dofs, vals = self.model.assemble_global_vec(
            self.model.avrx_dofs,
            self.model.svrx_nodes, 
            self.model.vrx_restraints
        )
        adofs = self.model.avrx_dofs.ravel()
        sdofs = dofs[vals.astype(bool)] # DOFs fijos
        fdofs = np.setdiff1d(adofs, sdofs) # DOFs libres
        
        return fdofs, sdofs

    
    def solve(self):
        """ Resuelve el problema y retorna resultados."""
        # Ensambla
        self.K0 = self.assemble_verax_K0()
        self.F = self.assemble_verax_F()
        free, supp = self.process_verax_restraints()
        
        # Reduce a DOFs libres    
        K0_ff = self.K0[np.ix_(free, free)]
        K0_sf = self.K0[np.ix_(supp, free)]
        F_f = self.F[free]
        F_s = self.F[supp]
        
        # Resuelve
        disps_f = cho_solve(cho_factor(K0_ff), F_f)
        
        self.disps = np.zeros(self.model.nvrx_dofs)
        self.disps[free] = disps_f
        self.react = K0_sf @ disps_f - F_s
        
        self.compute_internal_forces(self.disps)
        self.prepare_diagrams()
        return self

        
    
    def compute_internal_forces(self, glob_disps):
        """Calcula fuerzas internas verticales en elementos."""
        for elem in self.model.elements:
            elem.calculate_forces(glob_disps[elem.vrx_dofs])


    def prepare_diagrams(self, scales=(0.6, 0.6, 0.8, 0.6)):
        Fields = np.hstack([elem.get_fields() for elem in self.model.elements])
        x = Fields[0]

        norm = lambda vals, scale: np.vstack([x, vals/(np.max(np.abs(vals)) or 1.0) * scale, vals])
        sN, sV, sM, sd = scales

        self.diagrams     = norm(Fields[1], sN), norm(Fields[2], sV), norm(Fields[3], sM)
        self.deformations = norm(Fields[4], sd), norm(Fields[5], sd)

    def max_vals(self):
        return tuple(np.max(np.abs(d[2])) for d in (*self.diagrams, self.deformations[1]))

    def summary(self):
        maxN, maxV, maxM, maxw = self.max_vals()
        w = 48
        print(f"┌─ STATIC ANALYSIS {f'─'*(w-18)}┐")
        print(f"│ {'N max: ' + f'{maxN/1e3:.4f} kN':<{w-2}}│")
        print(f"│ {'V max: ' + f'{maxV/1e3:.4f} kN':<{w-2}}│")
        print(f"│ {'M max: ' + f'{maxM/1e3:.4f} kNm':<{w-2}}│")
        print(f"│ {'w max: ' + f'{maxw*1e3:.4f} mm':<{w-2}}│")
        print(f"└{f'─'*w}┘\n") 

    def plot(self):
        return plot_diagrams(self.diagrams, self.deformations)





