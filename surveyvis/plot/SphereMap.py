from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
import healpy as hp
import bokeh
import bokeh.plotting
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.visualization

from rubin_sim.utils.Site import Site
from rubin_sim.utils import calcLmstLast
from rubin_sim.utils import altAzPaFromRaDec, raDecFromAltAz
from rubin_sim.utils import approx_altAz2RaDec, approx_RaDec2AltAz
from rubin_sim.utils import ObservationMetaData

from surveyvis.plot.readjs import read_javascript

from surveyvis.sphere import offset_sep_bear, rotate_cart

ProjSliders = namedtuple("ProjSliders", ["alt", "az", "mjd"])

APPROX_COORD_TRANSFORMS = True

# Angles of excactly 90 degrees result in edge conditions, e.g.
# horizon circles with gaps depending on how trig is rounded.
# Define an "almost 90" to get consistent behaviour.
ALMOST_90 = np.degrees(np.arccos(0) - 2 * np.finfo(float).resolution)


class SphereMap:
    alt_limit = 0
    update_js_fname = "update_map.js"
    max_star_glyph_size = 15
    proj_slider_keys = ["mjd"]
    default_title = ""
    default_graticule_line_kwargs = {"color": "darkgray"}
    default_ecliptic_line_kwargs = {"color": "green"}
    default_galactic_plane_line_kwargs = {"color": "blue"}
    default_horizon_line_kwargs = {"color": "black", "line_width": 6}

    def __init__(self, plot=None, laea_limit_mag=88, mjd=None, site=Site("LSST")):
        """Base for maps of the sphere.

        Parameters
        ----------
        plot : `bokeh.plotting.figure.Figure`, optional
            Figure to which to add the map, by default None
        laea_limit_mag : `float`, optional
            Magnitude of the limit for Lamber azimuthal equal area plots,
            in degrees. By default 88.
        mjd : `float`, option
            The Modified Julian Date
        site : `rubin_sim.utils.Site.Site`, optional
            The site of the observatory, defaults to LSST.
        """

        self.laea_limit_mag = laea_limit_mag
        self.mjd = Time.now().mjd if mjd is None else mjd
        self.site = site

        if plot is None:
            self.plot = bokeh.plotting.figure(
                plot_width=512,
                plot_height=512,
                match_aspect=True,
                title=self.default_title,
            )
        else:
            self.plot = plot

        self.plot.axis.visible = False
        self.plot.grid.visible = False

        self.laea_proj = hp.projector.AzimuthalProj(rot=self.laea_rot, lamb=True)
        self.laea_proj.set_flip("astro")
        self.moll_proj = hp.projector.MollweideProj()
        self.moll_proj.set_flip("astro")

        self.figure = self.plot
        self.add_sliders()

    @property
    def lst(self):
        """Return the Local Sidereal Time."""
        lst = calcLmstLast(self.mjd, self.site.longitude_rad)[1] * 360.0 / 24.0
        return lst

    @lst.setter
    def lst(self, value):
        """Modify the MJD to match the LST, keeping the same (UT) day."""
        mjd_start = np.floor(self.mjd)
        lst_start = calcLmstLast(mjd_start, self.site.longitude_rad)[1] * 360.0 / 24.0
        self.mjd = mjd_start + ((value - lst_start) % 360) / 360.9856405809225

    @property
    def update_js(self):
        """Return javascript code to update the plots.

        Returns
        -------
        js_code : `str`
            Javascript code to update the bokeh model.
        """
        js_code = read_javascript(self.update_js_fname)
        return js_code

    @property
    def laea_rot(self):
        """Return the `rot` tuple to be used in the Lambert EA projection

        Returns
        -------
        rot : `tuple` [`float`]
            The `rot` tuple to be passed to `healpy.projector.AzimuthalProj`.
        """
        rot = (0, -90, 0) if self.site.latitude < 0 else (0, 90, 180)
        return rot

    @property
    def laea_limit(self):
        """Return the lat. furthest from the center for the LAEA projection.

        Returns
        -------
        `limit` : `float`
            The maximum (or minimum) value for the latitude shown in the
            Lambert Azimuthal Equal Area plot.
        """
        limit = (
            self.laea_limit_mag if self.site.latitude < 0 else -1 * self.laea_limit_mag
        )
        return limit

    def to_orth_zenith(self, hpx, hpy, hpz):
        """Convert healpy vector coordinates to orthographic coordinates

        Parameters
        ----------
        hpx : `numpy.ndarray`
            Healpy vector x coordinates
            x=1, y=0, z=0 corresponds to R.A.=0 deg, Decl=0 deg.
            x=-1, y=0, z=0 corresponds to R.A.=180 deg, Decl=0 deg.
        hpy : `numpy.ndarray`
            Healpy vector y coordinates
            x=0, y=1, z=0 corresponds to R.A.=90 deg, Decl=0 deg.
            x=0, y=-1, z=0 corresponds to R.A.=270 deg, Decl=0 deg.
        hpz : `numpy.ndarray`
            Healpy vector z coordinates
            x=0, y=0, z=1 corresponds to Decl=90 deg.
            x=0, y=0, z=-1 corresponds to Decl=-90 deg.

        Returns
        -------
        x : `numpy.ndarray`
            Orthographic x coordinate (positive to the right)
        y : `numpy.ndarray`
            Orthographic y coordinate (positive up)
        z : `numpy.ndarray`
            Orthographic z coordinate (positive toward the viewer)
        """
        x1, y1, z1 = rotate_cart(0, 0, 1, -90, hpx, hpy, hpz)
        x2, y2, z2 = rotate_cart(1, 0, 0, self.site.latitude + 90, x1, y1, z1)

        npole_x1, npole_y1, npole_z1 = rotate_cart(0, 0, 1, -90, 0, 0, 1)
        npole_x2, npole_y2, npole_z2 = rotate_cart(
            1, 0, 0, self.site.latitude + 90, npole_x1, npole_y1, npole_z1
        )
        x3, y3, z3 = rotate_cart(npole_x2, npole_y2, npole_z2, -self.lst, x2, y2, z2)

        # x3, y3, z3 have the center right, now rotate it so that north is "up"
        # the 2-3 transform is about the axis through the n pole, so
        # the n pole is the same in 3 an 2.

        # Find the direction of the north pole, angle form +x axis toward
        # +y axis
        npole_x3, npole_y3 = npole_x2, npole_y2
        orient = np.degrees(np.arctan2(npole_y3, npole_x3))

        # To the n pole on the y axis, we must rotate it the rest of the 90 deg
        x4, y4, z4 = rotate_cart(0, 0, 1, 90 - orient, x3, y3, z3)

        # In astronomy, we are looking out of the sphere from the center to the
        # back (which naturally results in west to the right).
        # Positive z is out of the screen behind us, and we are at the center,
        # so to visible part is when z is negative (coords[2]<=0).
        # So, set the points with positive z to NaN so they are
        # not shown, because they are behind the observer.

        # Use np.finfo(z3[0]).resolution instead of exactly 0, because the
        # assorted trig operations result in values slightly above or below
        # 0 when the horizon is in principle exactly 0, and this gives an
        # irregularly dotted/dashed appearance to the horizon if
        # a cutoff of exactly 0 is used.

        orth_invisible = z4 > np.finfo(z4.dtype).resolution
        x4[orth_invisible] = np.nan
        y4[orth_invisible] = np.nan
        z4[orth_invisible] = np.nan

        return x4, y4, z4

    def eq_to_horizon(self, ra, decl, degrees=True, cart=True):
        """Convert equatorial to horizon coordinates

        Parameters
        ----------
        ra : `numpy.ndarray`
            Values for Right Ascension
        decl : `numpy.ndarray`
            Values for declination
        degrees : bool, optional
            Values are in degrees (if False, values are in radians),
            by default True
        cart : bool, optional
            Return cartesion coordinates rather than alt, az, by default True

        Returns
        -------
        coords : `list` [`np.ndarray`]
            Either alt, az (if cart=False) with az measured east of north,
            or x, y with +x pointing west and +y pointing north

        Azimuth is east of north
        """

        # Due to a bug in altAzPaFromRaDec, it won't
        # work on pandas Series, so convert to a numpy
        # array if necessary.
        if isinstance(ra, pd.Series):
            ra = ra.values

        if isinstance(decl, pd.Series):
            decl = decl.values

        observation_metadata = ObservationMetaData(mjd=self.mjd, site=self.site)
        if degrees:
            ra_deg, decl_deg = ra, decl
        else:
            ra_deg, decl_deg = np.degrees(ra), np.degrees(decl)
        if APPROX_COORD_TRANSFORMS:
            alt, az = approx_RaDec2AltAz(
                ra_deg, decl_deg, self.site.latitude, self.site.longitude, self.mjd
            )
        else:
            alt, az, _ = altAzPaFromRaDec(ra_deg, decl_deg, observation_metadata)

        if cart:
            zd = np.pi / 2 - np.radians(alt)
            x = -zd * np.sin(np.radians(az))
            y = zd * np.cos(np.radians(az))

            invisible = alt < self.alt_limit
            x[invisible] = np.nan
            y[invisible] = np.nan
            return x, y

        if not degrees:
            return np.radians(alt), np.radians(az)

        return alt, az

    def make_healpix_data_source(self, hpvalues, nside=32, bound_step=1):
        """Make a data source of healpix values, corners, and projected coords.

        Parameters
        ----------
        hpvalues : `numpy.ndarray`
            Healpixel values (RING pixel ordering)
        nside : int, optional
            healpixel nside for display, by default 32
        bound_step : int, optional
            number of boundary points for each side of each healpixel,
            by default 1

        Returns
        -------
        hpix_datasource : `bokeh.models.ColumnDataSource`
            Data source for healpixel values and bounds.
        """
        values = np.copy(hpvalues)
        values[np.isnan(values)] = hp.UNSEEN
        values = hp.ud_grade(hpvalues, nside)
        values[values == hp.UNSEEN] = np.nan
        npix = hp.nside2npix(nside)
        npts = npix * 4 * bound_step
        hpids = np.arange(npix)
        hpix_bounds_vec = hp.boundaries(nside, hpids, bound_step)
        # Rearrange the axes to match what is used by hp.vec2ang
        hpix_bounds_vec_long = np.moveaxis(hpix_bounds_vec, 1, 2).reshape((npts, 3))
        ra, decl = hp.vec2ang(hpix_bounds_vec_long, lonlat=True)
        center_ra, center_decl = hp.pix2ang(nside, hpids, lonlat=True)
        x_hz, y_hz = self.eq_to_horizon(ra, decl)

        xs, ys, zs = self.to_orth_zenith(
            hpix_bounds_vec[:, 0, :], hpix_bounds_vec[:, 1, :], hpix_bounds_vec[:, 2, :]
        )

        x_laea, y_laea = self.laea_proj.vec2xy(hpix_bounds_vec_long.T)
        x_moll, y_moll = self.moll_proj.vec2xy(hpix_bounds_vec_long.T)

        # in hpix_bounds, each row corresponds to a healpixels, and columns
        # contain lists where elements of the lists correspond to corners.
        hpix_bounds = pd.DataFrame(
            {
                "hpid": hpids,
                "x_hp": hpix_bounds_vec[:, 0, :].tolist(),
                "y_hp": hpix_bounds_vec[:, 1, :].tolist(),
                "z_hp": hpix_bounds_vec[:, 2, :].tolist(),
                "ra": ra.reshape(npix, 4).tolist(),
                "decl": decl.reshape(npix, 4).tolist(),
                "x_orth": xs.tolist(),
                "y_orth": ys.tolist(),
                "z_orth": zs.tolist(),
                "x_laea": x_laea.reshape(npix, 4).tolist(),
                "y_laea": y_laea.reshape(npix, 4).tolist(),
                "x_moll": x_moll.reshape(npix, 4).tolist(),
                "y_moll": y_moll.reshape(npix, 4).tolist(),
                "x_hz": x_hz.reshape(npix, 4).tolist(),
                "y_hz": y_hz.reshape(npix, 4).tolist(),
            }
        )

        # in hpix_cornors, each row corresponds to one corner of one
        # healpix, identified by the hpid column.
        explode_cols = list(set(hpix_bounds.columns) - set(["hpid"]))
        hpix_corners = hpix_bounds.explode(column=explode_cols)

        # Hide points near the discontinuity at the pole in laea
        if self.site.latitude < 0:
            hide_laea = hpix_corners["decl"] > self.laea_limit
        else:
            hide_laea = hpix_corners["decl"] < self.laea_limit

        hpix_corners.loc[hide_laea, ["x_laea", "y_laea"]] = np.NaN

        # Hide points near the discontiuities at ra=180 in Mollweide
        resol = np.degrees(hp.nside2resol(nside))
        hide_moll = np.abs(hpix_corners["ra"] - 180) < (
            resol / np.cos(np.radians(decl))
        )
        hpix_corners.loc[hide_moll, ["x_moll", "y_moll"]] = np.NaN

        hpix_corners.replace([np.inf, -np.inf], np.NaN, inplace=True)
        hpix_data = hpix_corners.groupby("hpid").agg(lambda x: x.tolist())
        hpix_data["center_ra"] = center_ra
        hpix_data["center_decl"] = center_decl
        hpix_data["value"] = values

        values_are_finite = np.isfinite(values)
        finite_hpix_data = hpix_data.loc[values_are_finite, :]
        finite_hpids = hpids[values_are_finite]
        finite_values = values[values_are_finite]

        hpix = bokeh.models.ColumnDataSource(
            {
                "hpid": finite_hpids,
                "value": finite_values,
                "center_ra": finite_hpix_data["center_ra"].tolist(),
                "center_decl": finite_hpix_data["center_decl"].tolist(),
                "ra": finite_hpix_data["ra"].tolist(),
                "decl": finite_hpix_data["decl"].tolist(),
                "x_hp": finite_hpix_data["x_hp"].tolist(),
                "y_hp": finite_hpix_data["y_hp"].tolist(),
                "z_hp": finite_hpix_data["z_hp"].tolist(),
                "x_orth": finite_hpix_data["x_orth"].tolist(),
                "y_orth": finite_hpix_data["y_orth"].tolist(),
                "z_orth": finite_hpix_data["z_orth"].tolist(),
                "x_laea": finite_hpix_data["x_laea"].tolist(),
                "y_laea": finite_hpix_data["y_laea"].tolist(),
                "x_moll": finite_hpix_data["x_moll"].tolist(),
                "y_moll": finite_hpix_data["y_moll"].tolist(),
                "x_hz": finite_hpix_data["x_hz"].tolist(),
                "y_hz": finite_hpix_data["y_hz"].tolist(),
            }
        )

        return hpix

    def make_graticule_points(
        self,
        min_decl=-80,
        max_decl=80,
        decl_space=20,
        min_ra=0,
        max_ra=360,
        ra_space=30,
        step=1,
    ):
        """Create points that define graticules

        Parameters
        ----------
        min_decl : int, optional
            Decl. of minimum R.A. graticule
            and ends of declination graticules (deg),
            by default -80
        max_decl : int, optional
            Decl. of maximum R.A. graticulas
            and ends of declination graticules (deg),
            by default 80
        decl_space : int, optional
            Spacing of decl. graticules (deg), by default 20
        min_ra : int, optional
            R.A. of first R.A. graticule (deg), by default 0
        max_ra : int, optional
            R.A. of last R.A. graticule (deg), by default 360
        ra_space : int, optional
            Spacing of R.A. graticules (deg), by default 30
        step : int, optional
            Spacing of points along graticules, by default 1

        Returns
        -------
        graticule_points : `bokeh.models.ColumnDataSource`
            Bokeh data sources defining points in graticules.
        """

        # Bohek puts gaps in lines when there are NaNs in the
        # data frame. We will be platting many graticules, and we do not
        # want them connected, so define a "stop" element to separate
        # different graticules.
        stop_df = pd.DataFrame(
            {
                "decl": [np.nan],
                "ra": [np.nan],
                "grat": None,
                "x_orth": [np.nan],
                "y_orth": [np.nan],
                "z_orth": [np.nan],
                "x_laea": [np.nan],
                "y_laea": [np.nan],
                "x_moll": [np.nan],
                "y_moll": [np.nan],
                "x_hz": [np.nan],
                "y_hz": [np.nan],
            }
        )
        graticule_list = []

        # Define points in each declination graticule
        for decl in np.arange(min_decl, max_decl + decl_space, decl_space):
            ra_steps = np.arange(0, 360 + step)
            this_graticule = pd.DataFrame(
                {
                    "grat": f"decl{decl}",
                    "decl": decl,
                    "ra": ra_steps,
                    "x_hp": np.nan,
                    "y_hp": np.nan,
                    "z_hp": np.nan,
                    "x_orth": np.nan,
                    "y_orth": np.nan,
                    "z_orth": np.nan,
                    "x_laea": np.nan,
                    "y_laea": np.nan,
                    "x_moll": np.nan,
                    "y_moll": np.nan,
                    "x_hz": np.nan,
                    "y_hz": np.nan,
                }
            )
            this_graticule.loc[:, ["x_hp", "y_hp", "z_hp"]] = hp.ang2vec(
                this_graticule.ra, this_graticule.decl, lonlat=True
            )
            xs, ys, zs = self.to_orth_zenith(
                this_graticule.loc[:, "x_hp"],
                this_graticule.loc[:, "y_hp"],
                this_graticule.loc[:, "z_hp"],
            )
            this_graticule.loc[:, "x_orth"] = xs
            this_graticule.loc[:, "y_orth"] = ys
            this_graticule.loc[:, "z_orth"] = zs

            x_laea, y_laea = self.laea_proj.ang2xy(
                this_graticule["ra"], this_graticule["decl"], lonlat=True
            )
            this_graticule.loc[:, "x_laea"] = x_laea
            this_graticule.loc[:, "y_laea"] = y_laea

            x_moll, y_moll = self.moll_proj.ang2xy(
                this_graticule["ra"], this_graticule["decl"], lonlat=True
            )
            this_graticule.loc[:, "x_moll"] = x_moll
            this_graticule.loc[:, "y_moll"] = y_moll

            x_hz, y_hz = self.eq_to_horizon(
                this_graticule["ra"].values, this_graticule["decl"].values
            )
            this_graticule.loc[:, "x_hz"] = x_hz
            this_graticule.loc[:, "y_hz"] = y_hz

            graticule_list.append(this_graticule)
            graticule_list.append(stop_df)

        # Define points in each R.A. graticule
        for ra in np.arange(min_ra, max_ra + step, ra_space):
            decl_steps = np.arange(min_decl, max_decl + step, step)
            this_graticule = pd.DataFrame(
                {
                    "grat": f"ra{ra}",
                    "decl": decl_steps,
                    "ra": ra,
                    "x_hp": np.nan,
                    "y_hp": np.nan,
                    "z_hp": np.nan,
                    "x_orth": np.nan,
                    "y_orth": np.nan,
                    "z_orth": np.nan,
                    "x_laea": np.nan,
                    "y_laea": np.nan,
                    "x_moll": np.nan,
                    "y_moll": np.nan,
                    "x_hz": np.nan,
                    "y_hz": np.nan,
                }
            )
            this_graticule.loc[:, ["x_hp", "y_hp", "z_hp"]] = hp.ang2vec(
                this_graticule.ra, this_graticule.decl, lonlat=True
            )
            xs, ys, zs = self.to_orth_zenith(
                this_graticule.loc[:, "x_hp"],
                this_graticule.loc[:, "y_hp"],
                this_graticule.loc[:, "z_hp"],
            )
            this_graticule.loc[:, "x_orth"] = xs
            this_graticule.loc[:, "y_orth"] = ys
            this_graticule.loc[:, "z_orth"] = zs

            x_laea, y_laea = self.laea_proj.ang2xy(
                this_graticule["ra"], this_graticule["decl"], lonlat=True
            )
            this_graticule.loc[:, "x_laea"] = x_laea
            this_graticule.loc[:, "y_laea"] = y_laea

            x_moll, y_moll = self.moll_proj.ang2xy(
                this_graticule["ra"], this_graticule["decl"], lonlat=True
            )
            this_graticule.loc[:, "x_moll"] = x_moll
            this_graticule.loc[:, "y_moll"] = y_moll

            x_hz, y_hz = self.eq_to_horizon(
                this_graticule["ra"].values, this_graticule["decl"].values
            )
            this_graticule.loc[:, "x_hz"] = x_hz
            this_graticule.loc[:, "y_hz"] = y_hz

            graticule_list.append(this_graticule)
            graticule_list.append(stop_df)

        graticule_points = bokeh.models.ColumnDataSource(pd.concat(graticule_list))
        return graticule_points

    def make_horizon_graticule_points(
        self,
        min_alt=0,
        max_alt=80,
        alt_space=20,
        min_az=0,
        max_az=360,
        az_space=30,
        step=1,
    ):
        """Create points that define graticules

        Parameters
        ----------
        min_alt : int, optional
            Alt. of minimum az graticule
            and ends of alt graticules (deg),
            by default 0
        max_alt : int, optional
            Alt. of maximum az graticulas
            and ends of alt graticules (deg),
            by default 80
        alt_space : int, optional
            Spacing of alt. graticules (deg), by default 20
        min_az : int, optional
            Az of first azimuth graticule (deg), by default 0
        max_ra : int, optional
            Az of last azimuth graticule (deg), by default 360
        az_space : int, optional
            Spacing of azimuth graticules (deg), by default 30
        step : int, optional
            Spacing of points along graticules, by default 1

        Returns
        -------
        graticule_points : `bokeh.models.ColumnDataSource`
            Bokeh data sources defining points in graticules.
        """

        # Bohek puts gaps in lines when there are NaNs in the
        # data frame. We will be platting many graticules, and we do not
        # want them connected, so define a "stop" element to separate
        # different graticules.
        stop_df = pd.DataFrame(
            {
                "decl": [np.nan],
                "ra": [np.nan],
                "alt": [np.nan],
                "az": [np.nan],
                "grat": None,
                "x_orth": [np.nan],
                "y_orth": [np.nan],
                "z_orth": [np.nan],
                "x_laea": [np.nan],
                "y_laea": [np.nan],
                "x_moll": [np.nan],
                "y_moll": [np.nan],
                "x_hz": [np.nan],
                "y_hz": [np.nan],
            }
        )
        graticule_list = []

        # Define points in each alt graticule
        for alt in np.arange(min_alt, max_alt + alt_space, alt_space):
            radius = 90 - alt
            start_bear = 0
            end_bear = 360 + step
            this_graticule = pd.DataFrame(
                self.make_horizon_circle_points(
                    90, 0, radius, start_bear, end_bear, step
                ).data
            )
            this_graticule["grat"] = f"Alt{alt}"

            graticule_list.append(this_graticule)
            graticule_list.append(stop_df)

        for az in np.arange(min_az, max_az + step, az_space):
            radius = 90
            this_graticule = pd.DataFrame(
                self.make_horizon_circle_points(
                    0, az + 90, radius, 0, 360 + step, step
                ).data
            )
            this_graticule.query(
                f"(alt > {min_alt}) and (alt <= {max_alt}) and (abs(az-{az}) < 1)",
                inplace=True,
            )
            this_graticule.sort_values(by="alt", inplace=True)
            this_graticule["grat"] = f"Az{az}"

            graticule_list.append(this_graticule)
            graticule_list.append(stop_df)

        graticule_points = bokeh.models.ColumnDataSource(pd.concat(graticule_list))
        return graticule_points

    def make_circle_points(
        self,
        center_ra,
        center_decl,
        radius=90.0,
        start_bear=0,
        end_bear=360,
        step=1,
    ):
        """Create points along a circle or arc on a sphere

        Parameters
        ----------
        center_ra : `float`
            R.A. of the center of the circle (deg.).
        center_decl : `float`
            Decl. of the center of the circle (deg.).
        radius : float, optional
            Radius of the circle (deg.), by default 90.0
        start_bear : int, optional
            Bearing (E. of N.) of the start of the circle (deg.), by default 0
        end_bear : int, optional
            Bearing (E. of N.) of the end of the circle (deg.), by default 360
        step : int, optional
            Spacing of the points along the circle (deg.), by default 1

        Returns
        -------
        circle : `bokeh.models.ColumnDataSource`
            Bokeh data source for points in the circle.
        """
        ras = []
        decls = []
        bearings = []
        for bearing in range(start_bear, end_bear + step, step):
            ra, decl = offset_sep_bear(
                np.radians(center_ra),
                np.radians(center_decl),
                np.radians(radius),
                np.radians(bearing),
            )
            ras.append(np.degrees(ra))
            decls.append(np.degrees(decl))
            bearings.append(bearing)

        x0s, y0s, z0s = hp.ang2vec(np.array(ras), np.array(decls), lonlat=True).T
        xs, ys, zs = self.to_orth_zenith(x0s, y0s, z0s)

        x_laea, y_laea = self.laea_proj.ang2xy(
            np.array(ras), np.array(decls), lonlat=True
        )
        x_moll, y_moll = self.moll_proj.ang2xy(
            np.array(ras), np.array(decls), lonlat=True
        )
        x_hz, y_hz = self.eq_to_horizon(np.array(ras), np.array(decls))

        # Hide discontinuities
        if self.site.latitude < 0:
            laea_discont = np.array(decls) > self.laea_limit
        else:
            laea_discont = np.array(decls) < self.laea_limit
        x_laea[laea_discont] = np.nan
        y_laea[laea_discont] = np.nan

        moll_discont = np.abs(np.array(ras) - 180) < step
        x_moll[moll_discont] = np.nan
        y_moll[moll_discont] = np.nan

        circle = bokeh.models.ColumnDataSource(
            data={
                "bearing": bearings,
                "ra": ras,
                "decl": decls,
                "x_hp": x0s.tolist(),
                "y_hp": y0s.tolist(),
                "z_hp": z0s.tolist(),
                "x_orth": xs.tolist(),
                "y_orth": ys.tolist(),
                "z_orth": zs.tolist(),
                "x_laea": x_laea.tolist(),
                "y_laea": y_laea.tolist(),
                "x_moll": x_moll.tolist(),
                "y_moll": y_moll.tolist(),
                "x_hz": x_hz.tolist(),
                "y_hz": y_hz.tolist(),
            }
        )

        return circle

    def make_horizon_circle_points(
        self, alt=ALMOST_90, az=0, radius=90.0, start_bear=0, end_bear=360, step=1
    ):
        """Define points in a circle with the center defined in horizon coords.

        Parameters
        ----------
        alt : `float`
            Altitude of circle center, by default 90.
        az : `float`
            Azimuth of circle center, by default 0.
        radius : `float`, optional
            Radius of the circle (deg.), by default 90.0
        start_bear : int, optional
            Bearing of the start of the circle (deg.), by default 0
        end_bear : int, optional
            Bearing of the end of the circle (deg.), by default 360
        step : int, optional
            Spacing of points along the circle., by default 1

        Returns
        -------
        circle : `bokeh.models.ColumnDataSource`
            Bokeh data source with points along the circle.
        """
        observation_metadata = ObservationMetaData(mjd=self.mjd, site=self.site)
        if APPROX_COORD_TRANSFORMS:
            center_ra_arr, center_decl_arr = approx_altAz2RaDec(
                np.array([alt]),
                np.array([az]),
                self.site.latitude,
                self.site.longitude,
                self.mjd,
            )
            center_ra = center_ra_arr[0]
            center_decl = center_decl_arr[0]
        else:
            center_ra, center_decl = raDecFromAltAz(alt, az, observation_metadata)

        eq_circle_points = self.make_circle_points(
            center_ra, center_decl, radius, start_bear, end_bear, step
        )
        ra = np.array(eq_circle_points.data["ra"])
        decl = np.array(eq_circle_points.data["decl"])
        alt, az = self.eq_to_horizon(ra, decl, degrees=True, cart=False)

        circle_data = dict(eq_circle_points.data)
        circle_data["alt"] = alt.tolist()
        circle_data["az"] = az.tolist()

        circle = bokeh.models.ColumnDataSource(data=circle_data)

        return circle

    def make_points(self, points_data):
        """Create a bokes data source with locations of points on a sphere.

        Parameters
        ----------
        points_data : `Iterable` , `dict` , or `pandas.DataFrame`
            A source of data (to be passed to `pandas.DataFrame`)
            Must contain the following columns or keys:

            ``"ra"``
                The Right Ascension in degrees.
            ``"decl"``
                The declination in degrees.

        Returns
        -------
        point : `bokeh.models.ColumnDataSource`
            A data source with point locations, including projected coords.
        """

        points_df = pd.DataFrame(points_data)
        x0s, y0s, z0s = hp.ang2vec(points_df.ra, points_df.decl, lonlat=True).T
        xs, ys, zs = self.to_orth_zenith(x0s, y0s, z0s)

        x_laea, y_laea = self.laea_proj.ang2xy(
            points_df.ra, points_df.decl, lonlat=True
        )
        x_moll, y_moll = self.moll_proj.ang2xy(
            points_df.ra, points_df.decl, lonlat=True
        )
        x_hz, y_hz = self.eq_to_horizon(
            points_df.ra.values, points_df.decl.values, degrees=True, cart=True
        )

        # If point_df.ra and points_df.decl have only one value, ang2xy returns
        # scalars (or 0d arrays) not 1d arrays, but
        # bokeh.models.ColumnDataSource requires that column values
        # be python Sequences. So force results of ang2xy to be 1d arrays,
        # even when healpy returns 0d arrays.
        x_laea = x_laea.reshape(x_laea.size)
        y_laea = y_laea.reshape(y_laea.size)
        x_moll = x_moll.reshape(x_moll.size)
        y_moll = y_moll.reshape(y_moll.size)
        x_hz = x_hz.reshape(x_hz.size)
        y_hz = y_hz.reshape(y_hz.size)

        alt, az = self.eq_to_horizon(
            points_df.ra, points_df.decl, degrees=True, cart=False
        )
        invisible = alt < -1 * np.finfo(float).resolution
        x_hz[invisible] = np.nan
        y_hz[invisible] = np.nan

        data = {
            "name": points_df.name,
            "ra": points_df.ra.tolist(),
            "decl": points_df.decl.tolist(),
            "x_hp": x0s.tolist(),
            "y_hp": y0s.tolist(),
            "z_hp": z0s.tolist(),
            "x_orth": xs.tolist(),
            "y_orth": ys.tolist(),
            "z_orth": zs.tolist(),
            "x_laea": x_laea.tolist(),
            "y_laea": y_laea.tolist(),
            "x_moll": x_moll.tolist(),
            "y_moll": y_moll.tolist(),
            "x_hz": x_hz.tolist(),
            "y_hz": y_hz.tolist(),
            "glyph_size": points_df.glyph_size.tolist(),
        }

        # Add any additional data provided
        for column_name in points_df.columns:
            if column_name not in data.keys():
                data[column_name] = points_df[column_name].to_list()

        points = bokeh.models.ColumnDataSource(data=data)

        return points

    def make_marker_data_source(
        self,
        ra=None,
        decl=None,
        name="anonymous",
        glyph_size=5,
        min_mjd=None,
        max_mjd=None,
    ):
        """Add one or more circular marker(s) to the map.

        Parameters
        ----------
        ra : `float` or `Iterable`, optional
            R.A. of the marker (deg.), by default None
        decl : `float` or `Iterable`, optional
            Declination of the marker (deg.), by default None
        name : `str` or `Iterable` , optional
            Name for the thing marked, by default "anonymous"
        glyph_size : `int` or `Iterable`, optional
            Size of the marker, by default 5
        min_mjd : `float`
            Earlist time for which to show the marker
        max_mjd : `float`
            Latest time for which to show the marker

        Returns
        -------
        data_source : `bokeh.models.ColumnDataSource`
            A data source with marker locations, including projected coords.
        """
        ras = ra if isinstance(ra, Iterable) else [ra]
        decls = decl if isinstance(decl, Iterable) else [decl]
        if len(ras) > 0:
            glyph_sizes = (
                glyph_size
                if isinstance(glyph_size, Iterable)
                else [glyph_size] * len(ras)
            )
            names = [name] * len(ras) if isinstance(name, str) else name
        else:
            glyph_sizes = np.array([])
            names = np.array([])

        data = {
            "ra": ras,
            "decl": decls,
            "name": names,
            "glyph_size": glyph_sizes,
        }

        if (min_mjd is not None) or (max_mjd is not None):
            if len(ras) == 0:
                data["in_mjd_window"] = np.array([])
            else:
                data["in_mjd_window"] = [1] * len(ras)

        if min_mjd is not None:
            if not isinstance(min_mjd, Iterable):
                min_mjd = [min_mjd]
            if len(ras) < 1:
                min_mjd = np.array([])

            data["min_mjd"] = min_mjd

            for marker_index, this_min_mjd in enumerate(min_mjd):
                if self.mjd < this_min_mjd:
                    data["in_mjd_window"][marker_index] = 0

        if max_mjd is not None:
            if not isinstance(max_mjd, Iterable):
                max_mjd = [max_mjd]
            if len(ras) < 1:
                max_mjd = np.array([])

            data["max_mjd"] = max_mjd

            for marker_index, this_max_mjd in enumerate(max_mjd):
                if self.mjd > this_max_mjd:
                    data["in_mjd_window"][marker_index] = 0

        data_source = self.make_points(data)

        return data_source

    def add_sliders(self):
        """Add (already defined) sliders to the map."""
        self.sliders = OrderedDict()

    def add_mjd_slider(self):
        """Add a slider to control the MJD."""
        if "mjd" not in self.sliders:
            self.sliders["mjd"] = bokeh.models.Slider(
                start=self.mjd - 1,
                end=self.mjd + 1,
                value=self.mjd,
                step=1.0 / (24 * 12),
                title="MJD",
            )

            self.figure = bokeh.layouts.column(self.plot, *self.sliders.values())

    def set_js_update_func(self, data_source):
        """Set the javascript update functions for each slider

        Parameters
        ----------
        data_source : `bokeh.models.ColumnDataSource`
            The bokeh data source to update.
        """
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                center_alt_slider={"value": 90},
                center_az_slider={"value": 0},
                mjd_slider=self.sliders["mjd"],
                lat=self.site.latitude,
                lon=self.site.longitude,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass

    def show(self):
        """Show the map."""
        bokeh.io.show(self.figure)

    def add_healpix(self, data, cmap=None, nside=16, bound_step=1):
        """Add healpix values to the map

        Parameters
        ----------
        data : `numpy.ndarray`
            Healpixel values (RING pixel ordering)
        cmap : `bokeh.core.properties.ColorSpec`, optional
            _description_, by default None
        nside : `int`, optional
            Healpix nside to use for display, by default 16
        bound_step : `int`, optional
            number of boundary points for each side of each healpixel,
            by default 1

        Returns
        -------
        data_sounce : `bokeh.models.ColumnDataSource`
            The data source with the healpix values and bounds.
        cmap : `bokeh.core.properties.ColorSpec`
            The color map used
        hp_glype : `bokeh.models.glyphs.Patches`
            The bokeh glyphs for the plotted patches.
        """
        if isinstance(data, bokeh.models.DataSource):
            data_source = data
        else:
            data_source = self.make_healpix_data_source(data, nside, bound_step)

        self.healpix_data = data_source

        if cmap is None:
            cmap = make_zscale_linear_cmap(data_source.data["value"])

        self.healpix_cmap = cmap

        hpgr = self.plot.patches(
            xs=self.x_col,
            ys=self.y_col,
            fill_color=cmap,
            line_color=cmap,
            source=data_source,
        )

        self.healpix_glyph = hpgr.glyph
        self.healpix_renderer = hpgr

        hp_glyph = hpgr.glyph

        return data_source, cmap, hp_glyph

    def add_graticules(self, graticule_kwargs={}, line_kwargs={}):
        """Add graticules to the map

        Parameters
        ----------
        graticule_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_graticule_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        graticules : ` `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the graticules.
        """
        graticule_points = self.make_graticule_points(**graticule_kwargs)
        kwargs = deepcopy(self.default_graticule_line_kwargs)
        kwargs.update(line_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=graticule_points, **kwargs)
        return graticule_points

    def add_horizon_graticules(self, graticule_kwargs={}, line_kwargs={}):
        """Add graticules to the map

        Parameters
        ----------
        graticule_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_graticule_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        graticules : ` `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the graticules.
        """
        graticule_points = self.make_horizon_graticule_points(**graticule_kwargs)
        kwargs = deepcopy(self.default_graticule_line_kwargs)
        kwargs.update(line_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=graticule_points, **kwargs)
        return graticule_points

    def add_circle(self, center_ra, center_decl, circle_kwargs={}, line_kwargs={}):
        """Draw a circle on the map.

        Parameters
        ----------
        center_ra : `float`
            R.A. of the center of the circle (deg.)
        center_decl : `float`
            Decl. of the center of the circle (deg.)
        circle_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_circle_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        circle_points : `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the circle.
        """
        circle_points = self.make_circle_points(center_ra, center_decl, **circle_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=circle_points, **line_kwargs)
        return circle_points

    def add_horizon(
        self, zd=ALMOST_90, data_source=None, circle_kwargs={}, line_kwargs={}
    ):
        """Add a circle parallel to the horizon.

        Parameters
        ----------
        zd : int, optional
            Zenith distance of the circle (deg), by default (almost) 90
        data_source : `bokeh.models.ColumnDataSource`, optional
            Bokeh data source for points on the circle,
            None if the should be generated.
            By default, None
        circle_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_circle_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        circle_points : `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the circle.
        """
        if data_source is None:
            circle_points = self.make_horizon_circle_points(
                90, 0, radius=zd, **circle_kwargs
            )
            if "mjd" in self.sliders:
                self.set_js_update_func(circle_points)
        else:
            circle_points = data_source

        kwargs = deepcopy(self.default_horizon_line_kwargs)
        kwargs.update(line_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=circle_points, **kwargs)
        return circle_points

    def add_marker(
        self,
        ra=None,
        decl=None,
        name="anonymous",
        glyph_size=5,
        min_mjd=None,
        max_mjd=None,
        data_source=None,
        circle_kwargs={},
    ):
        """Add one or more circular marker(s) to the map.

        Parameters
        ----------
        ra : `float` or `Iterable`, optional
            R.A. of the marker (deg.), by default None
        decl : `float` or `Iterable`, optional
            Declination of the marker (deg.), by default None
        name : `str` or `Iterable` , optional
            Name for the thing marked, by default "anonymous"
        glyph_size : `int` or `Iterable`, optional
            Size of the marker, by default 5
        min_mjd : `float` or `Iterable`, optional
            Earliest time for which to show the marker.
        max_mjd : `float` or `Iterable`, optional
            Latest time for which to show the marker.
        data_source : `bokeh.models.ColumnDataSource`, optional
            Data source for the marker, None if a new one is to be generated.
            By default, None
        circle_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.circle``,
            by default {}

        Returns
        -------
        data_source : `bokeh.models.ColumnDataSource`
            A data source with marker locations, including projected coords.
        """
        if data_source is None:
            data_source = self.make_marker_data_source(
                ra, decl, name, glyph_size, min_mjd, max_mjd
            )

        self.plot.circle(
            x=self.x_col,
            y=self.y_col,
            size="glyph_size",
            source=data_source,
            **circle_kwargs,
        )

        return data_source

    def add_stars(
        self, points_data, data_source=None, mag_limit_slider=False, star_kwargs={}
    ):
        """Add stars to the map

        Parameters
        ----------
        points_data : `Iterable` , `dict` , or `pandas.DataFrame`
            A source of data (anything that can be passed to
            `pandas.DataFrame`)
            Must contain the following columns or keys:

            ``"ra"``
                The Right Ascension in degrees.
            ``"decl"``
                The declination in degrees.
        data_source : `bokeh.models.ColumnDataSource`, optional
            The bokeh data source to use (None to generate a new one).
            By default, None.
        mag_limit_slider : `bool` , optional
            Generate a slider limiting the magnitude of stars to plot,
            by default False
        star_kwargs : `dict` , optional
            _description_, by default {}

        Returns
        -------
        data_source : `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining star locations.
        """
        self.star_data = points_data
        if data_source is None:
            self.star_data_source = self.make_points(self.star_data)
        else:
            self.star_data_source = data_source

        self.plot.star(
            x=self.x_col,
            y=self.y_col,
            size="glyph_size",
            source=self.star_data_source,
            **star_kwargs,
        )

        if mag_limit_slider:
            mag_slider = bokeh.models.Slider(
                start=0,
                end=6.5,
                value=3,
                step=0.5,
                title="Magnitude limit for bright stars",
            )
            mag_slider.on_change("value", self.limit_stars)

            self.sliders["mag_limit"] = mag_slider

        self.figure = bokeh.layouts.column(self.plot, *self.sliders.values())

        return self.star_data_source

    def limit_stars(self, attr, old_limit, mag_limit):
        """Apply a magnitude limit to mapped stars

        Parameters
        ----------
        attr : `str`
            Attribute of the slider to use (ignored)
        old_limit : `float`
            Old value for the magnitude limit (ignored)
        mag_limit : `float`
            Now value for the magnitude limit

        Note
        ----
        This method is intended to be called as a callback by bokeh.
        """
        star_data = self.star_data.query(f"Vmag < {mag_limit}").copy()
        star_data.loc[:, "glyph_size"] = (
            self.max_star_glyph_size
            - (self.max_star_glyph_size / mag_limit) * star_data["Vmag"]
        )
        stars = self.make_points(star_data)
        self.star_data_source.data = dict(stars.data)

    def add_ecliptic(self, **kwargs):
        """Map the ecliptic.

        Returns
        -------
        points : `bokeh.models.ColumnDataSource`
            The bokeh data source with points on the ecliptic.
        """
        ecliptic_pole = SkyCoord(
            lon=0 * u.degree, lat=90 * u.degree, frame="geocentricmeanecliptic"
        ).icrs
        line_kwargs = deepcopy(self.default_ecliptic_line_kwargs)
        line_kwargs.update(kwargs)
        points = self.add_circle(
            ecliptic_pole.ra.deg, ecliptic_pole.dec.deg, line_kwargs=line_kwargs
        )
        return points

    def add_galactic_plane(self, **kwargs):
        """Map the galactic plane

        Returns
        -------
        points : `bokeh.models.ColumnDataSource`
            The bokeh data source with points on the galactic plane.
        """
        galactic_pole = SkyCoord(l=0 * u.degree, b=90 * u.degree, frame="galactic").icrs
        line_kwargs = deepcopy(self.default_galactic_plane_line_kwargs)
        line_kwargs.update(kwargs)
        points = self.add_circle(
            galactic_pole.ra.deg, galactic_pole.dec.deg, line_kwargs=line_kwargs
        )
        return points

    def decorate(self):
        """Add graticules, the ecliptic, and galactic plane to the map."""
        self.add_graticules()
        self.add_ecliptic()
        self.add_galactic_plane()


class Planisphere(SphereMap):
    x_col = "x_laea"
    y_col = "y_laea"
    default_title = "Planisphere"


class MollweideMap(SphereMap):
    x_col = "x_moll"
    y_col = "y_moll"
    default_title = "Mollweide"


class MovingSphereMap(SphereMap):
    def add_healpix(self, data, cmap=None, nside=16, bound_step=1):
        """Add healpix values to the map

        Parameters
        ----------
        data : `numpy.ndarray`
            Healpixel values (RING pixel ordering)
        cmap : `bokeh.core.properties.ColorSpec`, optional
            _description_, by default None
        nside : `int`, optional
            Healpix nside to use for display, by default 16
        bound_step : `int`, optional
            number of boundary points for each side of each healpixel,
            by default 1

        Returns
        -------
        data_sounce : `bokeh.models.ColumnDataSource`
            The data source with the healpix values and bounds.
        cmap : `bokeh.core.properties.ColorSpec`
            The color map used
        hp_glype : `bokeh.models.glyphs.Patches`
            The bokeh glyphs for the plotted patches.
        """
        data_source, cmap, hp_glyph = super().add_healpix(data, cmap, nside, bound_step)
        self.set_js_update_func(data_source)
        return data_source, cmap, hp_glyph

    def add_graticules(self, graticule_kwargs={}, line_kwargs={}):
        """Add graticules to the map

        Parameters
        ----------
        graticule_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_graticule_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        graticules : ` `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the graticules.
        """
        data_source = super().add_graticules(graticule_kwargs, line_kwargs)
        self.set_js_update_func(data_source)
        return data_source

    def add_circle(self, center_ra, center_decl, circle_kwargs={}, line_kwargs={}):
        """Draw a circle on the map.

        Parameters
        ----------
        center_ra : `float`
            R.A. of the center of the circle (deg.)
        center_decl : `float`
            Decl. of the center of the circle (deg.)
        circle_kwargs : dict, optional
            Keywords to be passed to ``SphereMap.make_circle_points``,
            by default {}
        line_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.line``,
            by default {}

        Returns
        -------
        circle_points : `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining the circle.
        """
        data_source = super().add_circle(
            center_ra, center_decl, circle_kwargs, line_kwargs
        )
        self.set_js_update_func(data_source)
        return data_source

    def add_stars(
        self, points_data, data_source=None, mag_limit_slider=False, star_kwargs={}
    ):
        """Add stars to the map

        Parameters
        ----------
        points_data : `Iterable` , `dict` , or `pandas.DataFrame`
            A source of data (anything that can be passed to
            `pandas.DataFrame`)
            Must contain the following columns or keys:

            ``"ra"``
                The Right Ascension in degrees.
            ``"decl"``
                The declination in degrees.
        data_source : `bokeh.models.ColumnDataSource`, optional
            The bokeh data source to use (None to generate a new one).
            By default, None.
        mag_limit_slider : `bool` , optional
            Generate a slider limiting the magnitude of stars to plot,
            by default False
        star_kwargs : `dict` , optional
            _description_, by default {}

        Returns
        -------
        data_source : `bokeh.models.ColumnDataSource`
            The bokeh data source with points defining star locations.
        """
        data_source = super().add_stars(
            points_data, data_source, mag_limit_slider, star_kwargs
        )
        self.set_js_update_func(data_source)
        return data_source

    def add_marker(
        self,
        ra=None,
        decl=None,
        name="anonymous",
        glyph_size=5,
        min_mjd=None,
        max_mjd=None,
        data_source=None,
        circle_kwargs={},
    ):
        """Add one or more circular marker(s) to the map.

        Parameters
        ----------
        ra : `float` or `Iterable`, optional
            R.A. of the marker (deg.), by default None
        decl : `float` or `Iterable`, optional
            Declination of the marker (deg.), by default None
        name : `str` or `Iterable` , optional
            Name for the thing marked, by default "anonymous"
        glyph_size : `int` or `Iterable`, optional
            Size of the marker, by default 5
        min_mjd : `float` or `Iterable`, optional
            Earliest time for which to show the marker.
        max_mjd : `float` or `Iterable`, optional
            Latest time for which to show the marker.
        data_source : `bokeh.models.ColumnDataSource`, optional
            Data source for the marker, None if a new one is to be generated.
            By default, None
        circle_kwargs : dict, optional
            Keywords to be passed to ``bokeh.plotting.figure.Figure.circle``,
            by default {}

        Returns
        -------
        data_source : `bokeh.models.ColumnDataSource`
            A data source with marker locations, including projected coords.
        """
        data_source = super().add_marker(
            ra,
            decl,
            name,
            glyph_size,
            min_mjd,
            max_mjd,
            data_source=data_source,
            circle_kwargs=circle_kwargs,
        )
        self.set_js_update_func(data_source)
        return data_source


class HorizonMap(MovingSphereMap):
    x_col = "x_hz"
    y_col = "y_hz"
    proj_slider_keys = ["mjd"]
    default_title = "Horizon"

    def set_js_update_func(self, data_source):
        """Set the javascript update functions for each slider

        Parameters
        ----------
        data_source : `bokeh.models.ColumnDataSource`
            The bokeh data source to update.
        """
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                mjd_slider=self.sliders["mjd"],
                lat=self.site.latitude,
                lon=self.site.longitude,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass

    def add_sliders(self, center_alt=90, center_az=0):
        """Add (already defined) sliders to the map."""
        super().add_sliders()
        self.sliders["mjd"] = bokeh.models.Slider(
            start=self.mjd - 1,
            end=self.mjd + 1,
            value=self.mjd,
            step=1.0 / (24 * 60),
            title="MJD",
        )

        self.figure = bokeh.layouts.column(self.plot, self.sliders["mjd"])


class ArmillarySphere(MovingSphereMap):
    x_col = "x_orth"
    y_col = "y_orth"
    proj_slider_keys = ["alt", "az", "mjd"]
    default_title = "Armillary Sphere"

    def set_js_update_func(self, data_source):
        """Set the javascript update functions for each slider

        Parameters
        ----------
        data_source : `bokeh.models.ColumnDataSource`
            The bokeh data source to update.
        """
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                center_alt_slider=self.sliders["alt"],
                center_az_slider=self.sliders["az"],
                mjd_slider=self.sliders["mjd"],
                lat=self.site.latitude,
                lon=self.site.longitude,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass

    def add_sliders(self, center_alt=90, center_az=180):
        """Add (already defined) sliders to the map."""
        super().add_sliders()
        self.sliders["alt"] = bokeh.models.Slider(
            start=-90,
            end=90,
            value=center_alt,
            step=np.pi / 180,
            title="center alt",
        )
        self.sliders["az"] = bokeh.models.Slider(
            start=-90, end=360, value=center_az, step=np.pi / 180, title="center Az"
        )
        self.sliders["mjd"] = bokeh.models.Slider(
            start=self.mjd - 1,
            end=self.mjd + 1,
            value=self.mjd,
            step=1.0 / (24 * 60),
            title="MJD",
        )

        self.figure = bokeh.layouts.column(
            self.plot, self.sliders["alt"], self.sliders["az"], self.sliders["mjd"]
        )


def make_zscale_linear_cmap(
    values, field_name="value", palette="Inferno256", *args, **kwargs
):
    zscale_interval = astropy.visualization.ZScaleInterval(*args, **kwargs)
    if np.any(np.isfinite(values)):
        scale_limits = zscale_interval.get_limits(values)
    else:
        scale_limits = [0, 1]
    cmap = bokeh.transform.linear_cmap(
        field_name, palette, scale_limits[0], scale_limits[1]
    )
    return cmap
