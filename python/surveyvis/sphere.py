import numpy as np


def offset_sep_bear(ra, decl, sep, bearing, degrees=False):
    """Calculate coordinates after an offset by a separation.

    Parameters
    ----------
    ra : `float`
       R.A. as a float in radians
    decl : `float`
       declination as a float in radians
    sep : `float`
       separation in radians
    bearing : `float`
       bearing (east of north) in radians
    degrees : `bool`
        arguments and returnes are in degrees (False for radians).

    Returns
    -------
    ra : `float`
       R.A. Right Ascension
    decl : `float`
       declination 

    """
    # Use cos formula:
    # cos(a)=cos(b)*cos(c)+sin(b)*sin(c)*cos(A)

    if degrees:
        ra = np.radians(ra)
        decl = np.radians(decl)
        sep = np.radians(sep)
        bearing = np.radians(bearing)

    np_sep = np.pi / 2 - decl

    new_np_sep = np.arccos(
        np.cos(np_sep) * np.cos(sep) + np.sin(np_sep) * np.sin(sep) * np.cos(bearing)
    )
    new_decl = np.pi / 2 - new_np_sep

    # use tan = sin/cos, sin rule to get sin, cos rule to get cos, cancel sin(np_sep) to avoid
    # problems when new_np_sep=90 deg.
    dra = np.arctan2(
        np.sin(sep) * np.sin(bearing) * np.sin(np_sep),
        np.cos(sep) - np.cos(new_np_sep) * np.cos(np_sep),
    )

    # Hack to match astropy behaviour at poles
    near_pole = np.abs(np.cos(decl)) < 1e-12
    if near_pole:
        dra = np.pi / 2 + np.cos(np_sep) * (np.pi / 2 - bearing)

    new_ra = ra + dra
    
    if degrees:
        new_ra = np.degrees(new_ra)
        new_decl = np.degrees(new_decl)

    return new_ra, new_decl


def horizon_to_eq(lat, alt, az, lst, degrees=False):
    """Convert horizon coordinates to equatorial coordinates.

    Parameters
    ----------
    lat : `float`
        Latitude of the observatory
    alt : `float`
        Altitude coordinate to transform
    az : `float`
        Azimuth coordinate to transform
    lst : `float`
        Local Sidereal Time
    degrees : bool, optional
        Unites are in degrees?, by default False

    Returns
    -------
    ra : `float`
        Right Ascension
    dec : `float`
        Declination
    """
    if degrees:
        lat = np.radians(lat)
        alt = np.radians(alt)
        az = np.radians(az)
        lst = np.radians(lst)

    decl = np.arcsin(np.sin(alt) * np.sin(lat) + np.cos(lat) * np.cos(alt) * np.cos(az))
    ha = np.arctan2(
        -1 * np.cos(alt) * np.cos(lat) * np.sin(az),
        np.sin(alt) - np.sin(lat) * np.sin(decl),
    )
    ra = lst - ha

    if degrees:
        ra = np.degrees(ra)
        decl = np.degrees(decl)

    return ra, decl


def eq_to_horizon(ra, decl, lat, lst, degrees=False):
    """Convert equatorial coordinates to horizon coordinates.

    Parameters
    ----------
    ra : `float`
        Right Ascension
    dec : `float`
        Declination
    lat : `float`
        Latitude of the observatory
    lst : `float`
        Local Sidereal Time
    degrees : `bool`, optional
        Unites are in degrees?, by default False

    Returns
    -------
    alt : `float`
        Altitude coordinate to transform
    az : `float`
        Azimuth coordinate to transform
    """
    if degrees:
        ra = np.radians(ra)
        decl = np.radians(decl)
        lat = np.radians(lat)
        lst = np.radians(lst)

    ha = lst - ra
    alt = np.arcsin(
        np.sin(decl) * np.sin(lat) + np.cos(decl) * np.cos(lat) * np.cos(ha)
    )
    az = np.arctan2(
        -1 * np.cos(decl) * np.cos(lat) * np.sin(ha),
        np.sin(decl) - np.sin(lat) * np.sin(alt),
    )

    if degrees:
        alt = np.degrees(alt)
        az = np.degrees(az)

    return alt, az

def rotate_cart(ux, uy, uz, angle, x0, y0, z0):
    """Rotate coordinates on a unit sphere around an axis

    Parameters
    ----------
    ux : `float`
        Input x coordinate
    uy : `float`
        Input y coordinate
    uz : `float`
        Input z coordinate
    angle : `float`
        Magnitude of the rotation.
    x0 : `float`
        x coordinate of a point on the axis of rotation
    y0 : `float`
        y coordinate of a point on the axis of rotation
    z0 : `float`
        z coordinate of a point on the axis of rotation
        
    Returns
    -------
    ux : `float`
        Output x coordinate
    uy : `float`
        Output y coordinate
    uz : `float`
        Output z coordinate
    """
    cosa = np.cos(np.radians(angle))
    ccosa = 1 - cosa
    sina = np.sin(np.radians(angle))
    rxx = cosa + ux * ux * ccosa
    rxy = ux * uy * ccosa - uz * sina
    rxz = ux * uz * ccosa + uy * sina
    ryx = uy * ux * ccosa + uz * sina
    ryy = cosa + uy * uy * ccosa
    ryz = uy * uz * ccosa - ux * sina
    rzx = uz * ux * ccosa - uy * sina
    rzy = uz * uy * ccosa + ux * sina
    rzz = cosa + uz * uz * ccosa
    x = rxx * x0 + rxy * y0 + rxz * z0
    y = ryx * x0 + ryy * y0 + ryz * z0
    z = rzx * x0 + rzy * y0 + rzz * z0
    return x, y, z
