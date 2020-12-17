from astropy import units as u
import numpy as np
from astropy.coordinates import Angle


def parangle(LST,RA,dec,lat,hang=None):
    """
    LST is an Angle -- the local sidereal time of the burst
    RA is an Angle -- right ascension of the source
    dec is an Angle -- declination of the source
    lat is an Angle -- latitude of the observatory
    """
    if hang!=None:
        ha_rad=hang
    else:
        ha = LST-RA
        if ha < Angle('0h'):
            ha=Angle('24h')+ha
        if ha>Angle('12h'):
            ha=ha-Angle('24h')


        ha+=Angle('24h')


        ha_rad = ha.to(u.radian).value
    dec_rad = dec.to(u.radian).value
    lat_rad = lat.to(u.radian).value
    print('ha',ha_rad)
    print('dec',dec_rad)
    print('lat',lat_rad)

    pa = np.arctan2(np.sin(ha_rad)*np.cos(lat_rad),(np.cos(dec_rad)*np.sin(lat_rad)-np.cos(lat_rad)*np.sin(dec_rad)*np.cos(ha_rad)))

    return pa*180/np.pi
