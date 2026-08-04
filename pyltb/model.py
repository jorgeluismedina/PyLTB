
import numpy as np
from .constructors import ElementFactory


class StabilityModel():
    def __init__(self):
        self.nvrx_dofn = 3 # numero de DOF por nodo (u, w, w,x) 
        self.nltr_dofn = 4 # numero de DOF por nodo (v, v,x, th, th,x)
        self.elements = []

        # estatico
        self.svrx_nodes = [] # tags de nodos con DOF de flex. vertical y desp. axial restringidos
        self.vrx_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # estabilidad
        self.sltr_nodes = [] # tags de nodos con DOF de flex. lateral y torsion restringidos
        self.ltr_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # apoyos elasticos laterales
        self.spring_nodes = []
        self.spring_pos = []
        self.spring_kv = [] # rigidez traslacional lateral (v)
        self.spring_kt = [] # rigidez torsional (theta)

        # cargas estatico
        self.loaded_nodes = [] # tags de nodos cargados
        self.nodal_loads = [] # cargas nodales
        self.loaded_elems = [] #tags

    def add_materials(self, materials): 
        self.materials = materials

    def add_sections(self, sections):
        self.sections = sections

    def assemble_global_vec(self, all_dofs, node_tag, values):
        dofs = all_dofs[node_tag].flatten()
        vals = np.array(values).flatten()
        return dofs, vals

    def add_nodes(self, coordinates):
        self.coords = coordinates
        self.nnodes = coordinates.shape[0]

        # DOF problema estatico (u, w, w,x)
        self.nvrx_dofs = self.nvrx_dofn * self.nnodes # numero total de DOF problema estatico
        self.avrx_dofs = np.arange(self.nvrx_dofs).reshape((self.nnodes, self.nvrx_dofn)) # DOF ordenados por nodo

        # DOF problema de estabilidad (v, v,x, th, th,x)
        self.nltr_dofs = self.nltr_dofn * self.nnodes # numero total de DOF problema estabilidad
        self.altr_dofs = np.arange(self.nltr_dofs).reshape((self.nnodes, self.nltr_dofn)) # DOF ordenados por nodo


    def build_node_alignments(self):
        """ Lista de alineacion de cada nodo, segun los elementos conectados. """
        self.node_align = ['centroid'] * self.nnodes
        visited = np.zeros(self.nnodes, dtype=bool)
        for elem in self.elements:
            for node in elem.conec:
                if not visited[node]:
                    self.node_align[node] = elem.align   # indexado por node id
                    visited[node] = True


    def add_uniform_elements(self, elements_data):
        """ Funcion solo para añadir elementos barra """
        for elem_data in elements_data:
            etype, mat_id, nodei, nodej = elem_data

            mat   = self.materials[int(mat_id)]
            sec   = self.sections[int(nodei)]
            conec = [int(nodei), int(nodej)]
            
            coords   = self.coords[conec]
            vrx_dofs = self.avrx_dofs[conec].flatten()
            ltr_dofs = self.altr_dofs[conec].flatten()

            elem = ElementFactory.create_uniform(etype, mat, sec, 
                                                 coords, conec, 
                                                 vrx_dofs, ltr_dofs)
            self.elements.append(elem)
        
        self.nelems = len(self.elements)
        self.build_node_alignments()
            

    def add_tapered_elements(self, elements_data, align=0):
        """
        Añade elementos de seccion variable.
 
        Parametros
        ----------
        elements_data : array-like — cada fila: [etype, mat_id, nodei, nodej]
        align : string
            Alineacion del eje de referencia local:
                0 → centroide G(x)    — sin acoplamiento axial-flexión (default)
                3 → fibra superior    — taper hacia abajo
                2 → fibra inferior    — taper hacia arriba
        """
        for elem_data in elements_data:
            etype, mat_id, nodei, nodej = elem_data

            mat   = self.materials[int(mat_id)]
            seci  = self.sections[int(nodei)]
            secj  = self.sections[int(nodej)]
            conec = [int(nodei), int(nodej)]
            
            coords   = self.coords[conec]
            vrx_dofs = self.avrx_dofs[conec].flatten()
            ltr_dofs = self.altr_dofs[conec].flatten()

            elem = ElementFactory.create_tapered(etype, mat, 
                                                 seci, secj,
                                                 coords, conec, 
                                                 vrx_dofs, ltr_dofs,
                                                 align=align)
            self.elements.append(elem)
        
        self.nelems = len(self.elements)
        self.build_node_alignments()


    def add_verax_restraints(self, verax_restraints_data):
        self.svrx_nodes     = list(verax_restraints_data[:,0].astype(int))
        self.vrx_restraints = verax_restraints_data[:,1:].astype(int)

    def add_lator_restraints(self, lator_restraints_data):
        self.sltr_nodes     = list(lator_restraints_data[:,0].astype(int))
        self.ltr_restraints = lator_restraints_data[:,1:].astype(int)

    def add_lateral_springs(self, springs_data):
        """
        Apoyos elasticos nodales en el problema lateral-torsional.
        Formato: [node, pos, kv, kdv, kt, kdt]
            kv  : rigidez traslacional lateral [F/L]  (v-DOF)
            kdv : rigidez curvatura lateral    [F]    (v'-DOF)
            kt  : rigidez torsional            [F·L]  (θ-DOF)
            kdt : rigidez warping torsional    [F]    (θ'-DOF)
            pos : pos. vertical, 0→G, 1→SC, 2→ala inf, 3→ala sup)
            * la posicion es solo para la traslacion lateral
        """
        self.spring_nodes = list(springs_data[:, 0].astype(int))
        self.spring_pos   = springs_data[:, 1].astype(int)
        self.spring_k_vec = springs_data[:, 2:].astype(float)


    def add_nodal_loads(self, nodal_loads_data):
        """
        Cargas puntuales nodales en coordenadas locales.
 
        Formato: [node, fxpos, fzpos, fxez, fzez, Fx, Fz, Mx]
            fzpos : altura de Fz — 0→G, 1→SC, 2→ala inf, 3→ala sup
                    (usado en el problema de estabilidad, StabilitySolver)
            fxpos : altura de Fx — mismos códigos
                    (la corrección ΔM = Fx·ez la aplica StaticSolver
                    en assemble_verax_F, con la geometría del elemento conectado)
            fxez, fzez : excentricidad de las cargas Fx y Fz respecto al eje de referencia local
            Fx, Fz, Mx : carga axial, vertical y momento nodal
        """
        self.loaded_nodes = list(nodal_loads_data[:,0].astype(int))
        self.nloads_pos   = nodal_loads_data[:,1:3].astype(int) # posiciones de Fx y Fz
        self.nloads_rez   = nodal_loads_data[:,3:5].astype(float) # z relativo a la pos. de Fx y Fz
        self.nodal_loads  = nodal_loads_data[:,5:].astype(float) # [Fx, Fz, Mx]
        

    def add_elem_loads(self, elem_loads_data):
        """
        Cargas distribuidas de elemento en coordenadas locales.
 
        Formato: [id_elem, qxpos, qzpos, qxez, qzez, qxi, qzi, qxj, qzj]
            qxpos : altura de qz — 0→G, 1→SC, 2→ala inf, 3→ala sup
            qzpos : altura de qx — mismos códigos
            qxi, qzi : intensidades en nodo i (axial, transversal)
            qxj, qzj : intensidades en nodo j (axial, transversal)
        """
        self.loaded_elems   = list(elem_loads_data[:,0].astype(int))
        self.eloads_pos = elem_loads_data[:,1:3].astype(int) # posiciones de qz y qx
        self.eloads_rez = elem_loads_data[:,3:5].astype(float) # z relativo a la pos. de qz y qx
        self.elem_loads = elem_loads_data[:,5:].astype(float)
        

        for load_data in elem_loads_data:
            id_elem = int(load_data[0])
            qxpos   = load_data[1].astype(int)
            qzpos   = load_data[2].astype(int)
            qxrz    = load_data[3].astype(float)
            qzrz    = load_data[4].astype(float)
            loads   = load_data[5:].astype(float)

            self.elements[id_elem].add_loads(qxpos, qzpos, qxrz, qzrz, *loads)

    def summary(self):
        L = float(self.coords[-1] - self.coords[0])
        w = 48
        print(f"\n┌─ MODEL {f'─'*(w-8)}┐")
        print(f"│ {'Length: ' + f'{L:.3f} m':<{w-2}} │")
        print(f"│ {'Nodes: ' + str(self.nnodes):<{w-2}} │")
        print(f"│ {'Elements: ' + str(self.nelems):<{w-2}} │")
        print(f"└{f'─'*w}┘\n")




        


