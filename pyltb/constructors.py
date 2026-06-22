from pyltb.elements.beamp import BeamP
from .elements.beamnp import BeamNP


class ElementFactory:
    """Factory para crear elementos"""
    
    registry = {}
    
    @classmethod
    def register(cls, name, element_class):
        """Registra nuevo tipo de elemento."""
        cls.registry[name] = element_class
        

    
    @classmethod
    def create_uniform(cls, etype, material, section, 
                       coord, conec, 
                       vrx_dofs, ltr_dofs):
        
        """Crea instancia del elemento."""
        if etype not in cls.registry:
            raise ValueError(f"Element type not supported: {etype}")
        
        return cls.registry[etype](material, section, 
                                   coord, conec, 
                                   vrx_dofs, ltr_dofs)
    

    @classmethod
    def create_tapered(cls, etype, material,
                       section_i, section_j,
                       coord, conec,
                       vrx_dofs, ltr_dofs,
                       align=0):
        
        """ Crea instancia de elemento tapered. """
        if etype not in cls.registry:
            raise ValueError(f"Element type not supported: {etype}")
        
        return cls.registry[etype](material, 
                                   section_i, section_j, 
                                   coord, conec, 
                                   vrx_dofs, ltr_dofs,
                                   align=align)
        


# Registrar elemento disponible
ElementFactory.register(0, BeamP)
ElementFactory.register(1, BeamNP)



