
import numpy as np
from scipy.linalg import cho_factor, cho_solve



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
                fxez   = sec.z_from_ref(align, pos) + rez
                  
                F[dof_Mx] -= Fx * fxez #el trabajo de la fuerza axial tiene que ser negativo

        if self.model.loaded_elems:
            for id_elem in self.model.loaded_elems:
                F[self.model.elements[id_elem].vrx_dofs] += self.model.elements[id_elem].loads

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
        self.fields = [elem.get_fields() for elem in self.model.elements]
        
    
    def compute_internal_forces(self, glob_disps):
        """Calcula fuerzas internas verticales en elementos."""
        for elem in self.model.elements:
            elem.calculate_forces(glob_disps[elem.vrx_dofs])


    
    def max_vals(self):
        """Valores máximos de fuerzas internas y desplazamiento vertical."""
        N = np.concatenate([f[1] for f in self.fields])
        V = np.concatenate([f[2] for f in self.fields])
        M = np.concatenate([f[3] for f in self.fields])
        w = self.disps.reshape(self.model.nnodes, self.model.nvrx_dofn)[:, 1]
        return (N[np.argmax(np.abs(N))],
                V[np.argmax(np.abs(V))],
                M[np.argmax(np.abs(M))],
                w[np.argmax(np.abs(w))])

    
    
    def prepare_diagrams(self, esc1=0.6, esc2=0.8, esc3=0.6):
        """Diagrama listo para plotear, sin args externos."""
        all_N = np.concatenate([f[1] for f in self.fields])
        all_V = np.concatenate([f[2] for f in self.fields])
        all_M = np.concatenate([f[3] for f in self.fields])
        all_u = np.concatenate([f[4] for f in self.fields])
        all_w = np.concatenate([f[5] for f in self.fields])

        max_N   = np.max(np.abs(all_N)) or 1
        max_V   = np.max(np.abs(all_V)) or 1
        max_M   = np.max(np.abs(all_M)) or 1
        max_def = max(np.max(np.abs(all_u)), np.max(np.abs(all_w))) or 1

        N_globals, V_globals, M_globals, def_shapes = [], [], [], []

        for elem, (x, N, V, M, u, w) in zip(self.model.elements, self.fields):
            X = elem.coords[0] + x
            N_globals.append(np.vstack([X, N/max_N*esc1, N]))
            V_globals.append(np.vstack([X, V/max_V*esc1, V]))
            M_globals.append(np.vstack([X, M/max_M*esc2, M]))
            def_shapes.append(np.vstack([X + u/max_def*esc3, w/max_def*esc3, u, w]))

        return N_globals, V_globals, M_globals, def_shapes





