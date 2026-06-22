
import numpy as np
from itertools import product

def gauss_1d(npts):
    """ Puntos de gauss en dominio unidimensional para el dominio [0,1]"""
    if npts == 1:
        pts = [0.5]
        wts = [1.0]

    elif npts == 2:
        pts = [0.211324865405187117, 0.788675134594812882]
        wts = [0.5, 0.5]

    elif npts == 3:
        pts = [0.112701665379258311, 0.5, 0.887298334620741688]
        wts = [0.777777777777777778, 0.444444444444444444,
               0.777777777777777778]   
        
    elif npts == 4:
        pts = [0.069431844202973712, 0.330009478207571867,
               0.669990521792428132, 0.930568155797026287]
        wts = [0.173927422568726928, 0.326072577431273071,
               0.326072577431273071, 0.173927422568726928]
        
        
    else:
        msg = "The number of points should be in [2, 10]"
        raise ValueError(msg)

    return pts, wts





def gauss_nd(npts, ndim):
    """
    Returns points and weights for Gauss quadrature in
    an ND hypercube using products from one-dimensional quadrature scheme.

    Parameters
    ----------
    npts : (int) : Number of sample points.

    Returns
    -------
    nd_wts : (ndarray) : Weights for the Gauss-Legendre quadrature.
    nd_pts : (ndarray) : Points for the Gauss-Legendre quadrature.
    """
    pts, wts = gauss_1d(npts)
    nd_pts = np.array(list(product(pts, repeat=ndim)))
    nd_wts = np.prod(np.array(list(product(wts, repeat=ndim))), axis=1)
    return nd_pts, nd_wts