import numpy as np

def radiometer(tsamp, bw, npol, SEFD):
    """ 
    radiometer(tsamp, bw, npol, Tsys, G):                                                                                                                                 tsamp is the time resolution in milliseconds                                                                                                                          bw is the bandwidth in MHz                                                                                                                                            npol is the number of polarizations                                                                                                                               
    Tsys is the system temperature in K (typical value for Effelsberg = 20K)                                                                                          
    G is the gain
    """

    return (SEFD)*(1/np.sqrt((bw*1.e6)*npol*tsamp*1e-3))
