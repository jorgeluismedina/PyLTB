
from pyltb.material import Material
from pyltb.sections.section_ms import ISection_MS
from pyltb.sections.section_utils import interpolate_multiple_sections
from pyltb.gauss_quad import gauss_1d


# Materiales
material1 = Material(E=2.1e11, nu=0.2, rho=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_MS(h=0.3, bf1=0.20, bf2=0.20, tw=0.01, tf1=0.015, tf2=0.015, r1=0.0, r2=0.0) #[m]
sect2 = ISection_MS(h=0.2, bf1=0.15, bf2=0.15, tw=0.01, tf1=0.015, tf2=0.015, r1=0.0, r2=0.0) #[m]

sect1.summary()


points, weights = gauss_1d(3)

sections = interpolate_multiple_sections(sect1, sect2, points)

for sec in sections:
    print(sec.h)
    print(sec.bf1)
    print(sec.bf2)
    print(sec.It)
    print('\n')


