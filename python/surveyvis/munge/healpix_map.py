import numpy as np
import pandas as pd
import healpy as hp

def create_healpix_df(nside, bound_step=1, lat=[32]):
    """Create a DataFrame with parameters for healpixels.
    
    Parameters
    ----------
    nside : `int`
        The nside for the healpix map.
    bound_step : `int`
        The bound step for perimeters of healpixels. (See healpy.boundaries)
    lat: `float`
        The latitude of the observatory, in decimal degrees.
        
    Returns
    -------
    healpix_id : `pandas.DataFrame`
        A DataFrame with columns describing healpixel.
    """

    npix = hp.nside2npix(nside)
    npts = npix * 4 * bound_step
    hpids = np.arange(npix)
    hpix_bounds_vec = hp.boundaries(nside, hpids, bound_step)
    # Rearrange the axes to match what is used by hp.vec2ang
    hpix_bounds_vec_long = np.moveaxis(hpix_bounds_vec, 1, 2).reshape((npts, 3))
    ra, decl = hp.vec2ang(hpix_bounds_vec_long, lonlat=True)

    xs, ys, zs = self.to_orth_zenith(
        hpix_bounds_vec[:, 0, :],
        hpix_bounds_vec[:, 1, :],
        hpix_bounds_vec[:, 2, :]
    )

    x_laea, y_laea = self.laea_proj.vec2xy(hpix_bounds_vec_long.T)
    x_moll, y_moll = self.moll_proj.vec2xy(hpix_bounds_vec_long.T)

    # in hpix_bounds, each row corresponds to a healpixes, and columns
    # contain lists where elements of the lists correspond to corners.
    hpix_bounds = pd.DataFrame(
        {
            "hpid": hpids,
            "x0": hpix_bounds_vec[:, 0, :].tolist(),
            "y0": hpix_bounds_vec[:, 1, :].tolist(),
            "z0": hpix_bounds_vec[:, 2, :].tolist(),
            "ra": ra.reshape(npix, 4).tolist(),
            "decl": decl.reshape(npix, 4).tolist(),
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": zs.tolist(),
            "x_laea": x_laea.reshape(npix, 4).tolist(),
            "y_laea": y_laea.reshape(npix, 4).tolist(),
            "x_moll": x_moll.reshape(npix, 4).tolist(),
            "y_moll": y_moll.reshape(npix, 4).tolist(),
        }
    )
 
    # in hpix_cornors, each row corresponds to one corner of one
    # healpix, identified by the hpid column.
    explode_cols = list(set(hpix_bounds.columns) - set(["hpid"]))
    hpix_corners = hpix_bounds.explode(column=explode_cols)

    # Hide points near the discontinuity at the pole in laea
    if self.lat<0:
        hide_laea = hpix_corners["decl"] > self.laea_limit
    else:
        hide_laea = hpix_corners["decl"] < self.laea_limit

    hpix_corners.loc[hide_laea, ["x_laea", "y_laea"]] = np.NaN

    # Hide points near the discontiuities at ra=180 in Mollweide
    resol = np.degrees(hp.nside2resol(nside))
    hide_moll = np.abs(hpix_corners["ra"] - 180) < (resol / np.cos(np.radians(decl)))
    hpix_corners.loc[hide_moll, ["x_moll", "y_moll"]] = np.NaN

    # Hide points behind us in the orthographic projection
    hide_ortho = hpix_corners["z"] < 0
    hpix_corners.loc[hide_ortho, ["x"]] = np.NaN

    hpix_corners.replace([np.inf, -np.inf], np.NaN, inplace=True)

    df = pd.DataFrame(
        {
            "hpid": finite_hpids,
            "ra": finite_hpix_data['ra'].tolist(),
            "decl": finite_hpix_data["decl"].tolist(),
            "x0": finite_hpix_data["x0"].tolist(),
            "y0": finite_hpix_data["y0"].tolist(),
            "z0": finite_hpix_data["z0"].tolist(),
            "x": finite_hpix_data["x"].tolist(),
            "y": finite_hpix_data["y"].tolist(),
            "z": finite_hpix_data["z"].tolist(),
            "x_laea": finite_hpix_data["x_laea"].tolist(),
            "y_laea": finite_hpix_data["y_laea"].tolist(),
            "x_moll": finite_hpix_data["x_moll"].tolist(),
            "y_moll": finite_hpix_data["y_moll"].tolist(),
        }
    )

    return df