
import numpy as np
import scipy as sp


class StabilitySolver():
    def __init__(self, model):
        self.model = model

    def assemble_lator_K0(self):
        """Ensambla matriz de rigidez global."""
        nltr_dofs = self.model.nltr_dofs
        K0_ltr = np.zeros((nltr_dofs, nltr_dofs))
        for elem in self.model.elements:
            K0_ltr[np.ix_(elem.ltr_dofs, elem.ltr_dofs)] += elem.K0_ltr

        for i, node in enumerate(self.model.spring_nodes):
            dof_v  = self.model.altr_dofs[node, 0]  # DOF v
            dof_dv = self.model.altr_dofs[node, 1]  # DOF v'
            dof_t  = self.model.altr_dofs[node, 2]  # DOF theta
            dof_dt = self.model.altr_dofs[node, 3]  # DOF theta'

            kv  = self.model.spring_k_vec[i, 0]
            kdv = self.model.spring_k_vec[i, 1]
            kt  = self.model.spring_k_vec[i, 2]
            kdt = self.model.spring_k_vec[i, 3]

            pos = self.model.spring_pos[i]
            sec = self.model.sections[node]
            ez  = sec.z_from_ref(self.model.node_align[node], pos)

            K0_ltr[dof_v,  dof_v]  += kv
            K0_ltr[dof_v,  dof_t]  -= kv * ez   # acoplamiento (estudiar mejor el cambio de signo)
            K0_ltr[dof_t,  dof_v]  -= kv * ez   # acoplamiento (estudiar mejor el cambio de signo)
            K0_ltr[dof_t,  dof_t]  += kv * ez**2 + kt
            K0_ltr[dof_dv, dof_dv] += kdv       # v'
            K0_ltr[dof_dt, dof_dt] += kdt       # θ'
                    
        return K0_ltr

    def assemble_lator_Kg(self):
        """Ensambla matriz geometrica global."""
        nltr_dofs = self.model.nltr_dofs
        Kg_ltr = np.zeros((nltr_dofs, nltr_dofs))
        for elem in self.model.elements:
            elem.update_lator_Kg()
            Kg_ltr[np.ix_(elem.ltr_dofs, elem.ltr_dofs)] += elem.Kg_ltr

        for i, node in enumerate(self.model.loaded_nodes):
            dof_t = self.model.altr_dofs[node, 2]      # DOF θ del nodo
            Fz    = self.model.nodal_loads[i, 1]       # carga vertical

            pos  = self.model.nodal_loads_pos[i, 1]    # código de altura
            rez  = self.model.nodal_loads_rez[i, 1]    # z relativo a la posición de la carga
            sec  = self.model.sections[node]
            fzez = sec.z_from_ref(1, pos) + rez
            
            Kg_ltr[dof_t, dof_t] += Fz * fzez
         
        return Kg_ltr
    
    def process_lator_restraints(self):
        """Separa DOFs fijos y libres."""
        dofs, vals = self.model.assemble_global_vec(
            self.model.altr_dofs,
            self.model.sltr_nodes, 
            self.model.ltr_restraints
        )
        adofs = self.model.altr_dofs.ravel()
        sdofs = dofs[vals.astype(bool)] # DOFs fijos
        fdofs = np.setdiff1d(adofs, sdofs) # DOFs libres
        
        return fdofs, sdofs
    
    def solve(self, nmodes=5):
        """ Resuelve el problema de estabilidad y retorna resultados."""
        # Ensambla
        self.K0 = self.assemble_lator_K0()
        self.Kg = self.assemble_lator_Kg()
        free, supp = self.process_lator_restraints()
        
        # Reduce a DOFs libres    
        K0_ff = self.K0[np.ix_(free, free)]
        Kg_ff = self.Kg[np.ix_(free, free)]
        
        # Resuelve autovectores y autovalores 
        # Problema invertido: -Kg φ = λ K0 φ, con μ = 1/λ la carga crítica de pandeo
        # eigh ordena λ ascendente → los k mayores λ son los k menores μ
        # los autovectores modes son columnas, cada columna es un modo de pandeo
        n = free.size
        k = min(nmodes, n)
        lam, modes = sp.linalg.eigh(-Kg_ff, K0_ff, subset_by_index=[n - k, n - 1])

        # λ descendente → μ ascendente; descarta λ ≤ 0 (pandeo con carga invertida)
        lam, modes = lam[::-1], modes[:, ::-1]
        pos = lam > 1e-12
        self.mu_crs = 1 / lam[pos]

        # Reconstruccion de modos completos con apoyos incluidos
        self.modes = np.zeros((self.model.nltr_dofs, self.mu_crs.size))
        self.modes[free, :] = modes[:, pos]

        self.transform_modes_to_S()
        return self

    def transform_modes_to_S(self):
        """
        Convierte self.modes (DOFs centroidales) a self.modes_S (DOFs en el centro de corte).
        """
        n_nodes = self.model.nnodes

        # Arrays con los índices de cada DOF por nodo
        dof_v  = self.model.altr_dofs[:, 0]   # shape (n_nodes,)
        dof_dv = self.model.altr_dofs[:, 1]
        dof_t  = self.model.altr_dofs[:, 2]
        dof_dt = self.model.altr_dofs[:, 3]

        # Vector de excentricidades zS en cada nodo
        zS = np.array([self.model.sections[n].z_from_ref(self.model.node_align[n], 1) 
                       for n in range(n_nodes)])

        # Copia inicial de los modos (el giro y su derivada no cambian)
        modes_S = self.modes.copy()
        modes_S[dof_v, :]  -= zS[:, None] * self.modes[dof_t, :]  # Transformación v  : v_S = v_C - zS * θ
        modes_S[dof_dv, :] -= zS[:, None] * self.modes[dof_dt, :] # Transformación v' : v'_S = v'_C - zS * θ'
        
        self.modes_S = modes_S

    def summary(self, n=1, ref=None):
        w = 48
        print(f"┌─ STABILITY ANALYSIS {f'─'*(w-21)}┐")
        for i, mu in enumerate(self.mu_crs[:n]):
            print(f"│ {'Mode ' + str(i+1) + ': μ_cr = ' + f'{mu:.4f}':<{w-2}} │")
            if ref and i == 0:
                max_name_len = max(len(name) for name in ref.keys())
                for name, val in ref.items():
                    delta = abs(mu - val) / val * 100
                    formatted_name = f"{name:<{max_name_len}}"
                    content = f"{formatted_name}  {val:.4f}  (Δ = {delta:.2f}%)"
                    print(f"│       {content:<{w-8}} │")
        print(f"└{f'─'*w}┘\n")

    def plot(self, imode=0, scale=1.0, n_sec=2, curves=True):
        from pyltb.plotting import plot_buckling_mode
        return plot_buckling_mode(self.model, self.mu_crs, self.modes_S, imode, scale, n_sec, curves)