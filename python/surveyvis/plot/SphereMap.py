from collections import OrderedDict, namedtuple
from copy import deepcopy
from xml.etree.ElementTree import ProcessingInstruction

import numpy as np
import pandas as pd
import healpy as hp
import bokeh
import bokeh.plotting
import astropy.units as u
from astropy.coordinates import SkyCoord

from surveyvis.plot.readjs import read_javascript

from surveyvis.sphere import (
    offset_sep_bear,
    horizon_to_eq,
    eq_to_horizon,
    rotate_cart
)

ProjSliders = namedtuple("ProjSliders", ["alt", "az", "lst"])

class SphereMap:
    alt_limit = 0
    update_js_fname = 'update_map.js'
    max_star_glyph_size = 15
    proj_slider_keys = ['lst']
    default_title = ''
    default_graticule_line_kwargs = {'color': 'darkgray'}
    default_ecliptic_line_kwargs = {'color': 'green'}
    default_galactic_plane_line_kwargs = {'color': 'blue'}
    default_horizon_line_kwargs = {"color": "black", "line_width": 6}

    def __init__(self, plot=None, lst=0.0, lat=-30.244639, laea_limit_mag=88):
        
        self.lat = lat
        self.lst = lst
        self.laea_limit_mag = laea_limit_mag

        if plot is None:
            self.plot = bokeh.plotting.figure(
                plot_width=512, plot_height=512, match_aspect=True, title=self.default_title,
            )
        else:
            self.plot = plot

        self.plot.axis.visible = False
        self.plot.grid.visible = False

        self.laea_proj = hp.projector.AzimuthalProj(rot=self.laea_rot, lamb=True)
        self.laea_proj.set_flip('astro')
        self.moll_proj = hp.projector.MollweideProj()
        self.moll_proj.set_flip('astro')

        self.figure = self.plot
        self.add_sliders()


    @property
    def update_js(self):
        js_code = read_javascript(self.update_js_fname)
        return js_code

    @property
    def laea_rot(self):
        rot = (0, -90, 0) if self.lat<0 else (0, 90, 180)
        return rot

    @property
    def laea_limit(self):
        limit = self.laea_limit_mag if self.lat<0 else -1*self.laea_limit_mag
        return limit

    def to_orth_zenith(self, hpx, hpy, hpz):
        x1, y1, z1 = rotate_cart(0, 0, 1, -90, hpx, hpy, hpz)
        x2, y2, z2 = rotate_cart(1, 0, 0, self.lat+90, x1, y1, z1)
          
        orth_invisible = z2 > 0
        x2[orth_invisible] = np.nan
        y2[orth_invisible] = np.nan
        z2[orth_invisible] = np.nan
        return x2, y2, z2

    def eq_to_horizon(self, ra, decl, degrees=True, cart=True):
        alt, az = eq_to_horizon(ra, decl, self.lat, self.lst, degrees=degrees)
        if cart:
            x, y = eq_to_horizon(ra, decl, self.lat, self.lst, degrees=degrees, cart=cart)
            invisible = alt < self.alt_limit
            x[invisible] = np.nan
            y[invisible] = np.nan
            return x, y
        else:
            return alt, az

    def make_healpix_data_source(self, hpvalues, nside=32, bound_step=1):
        values = hp.ud_grade(hpvalues, nside)
        npix = hp.nside2npix(nside)
        npts = npix * 4 * bound_step
        hpids = np.arange(npix)
        hpix_bounds_vec = hp.boundaries(nside, hpids, 1)
        # Rearrange the axes to match what is used by hp.vec2ang
        hpix_bounds_vec_long = np.moveaxis(hpix_bounds_vec, 1, 2).reshape((npts, 3))
        ra, decl = hp.vec2ang(hpix_bounds_vec_long, lonlat=True)
        center_ra, center_decl = hp.pix2ang(nside, hpids, lonlat=True)
        x_hz, y_hz = self.eq_to_horizon(ra, decl)

        xs, ys, zs = self.to_orth_zenith(
            hpix_bounds_vec[:, 0, :],
            hpix_bounds_vec[:, 1, :],
            hpix_bounds_vec[:, 2, :]
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
        hide_ortho = hpix_corners["z_orth"] < 0
        hpix_corners.loc[hide_ortho, ["x_orth"]] = np.NaN

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
                "ra": finite_hpix_data['ra'].tolist(),
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
        self, min_decl=-80, max_decl=80, decl_space=20, min_ra=0, max_ra=360, ra_space=30, step=1,
    ):
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
                'x_hz': [np.nan],
                'y_hz': [np.nan],
            }
        )
        graticule_list = []

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
                this_graticule.loc[:, 'x_hp'],
                this_graticule.loc[:, 'y_hp'],
                this_graticule.loc[:, 'z_hp']
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
            
            x_hz, y_hz = self.eq_to_horizon(this_graticule["ra"], this_graticule["decl"])
            this_graticule.loc[:, "x_hz"] = x_hz
            this_graticule.loc[:, "y_hz"] = y_hz

            graticule_list.append(this_graticule)
            graticule_list.append(stop_df)

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
                this_graticule.loc[:, 'x_hp'],
                this_graticule.loc[:, 'y_hp'],
                this_graticule.loc[:, 'z_hp']
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

            x_hz, y_hz = self.eq_to_horizon(this_graticule["ra"], this_graticule["decl"])
            this_graticule.loc[:, "x_hz"] = x_hz
            this_graticule.loc[:, "y_hz"] = y_hz

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

        x_laea, y_laea = self.laea_proj.ang2xy(np.array(ras), np.array(decls), lonlat=True)
        x_moll, y_moll = self.moll_proj.ang2xy(np.array(ras), np.array(decls), lonlat=True)
        x_hz, y_hz = self.eq_to_horizon(np.array(ras), np.array(decls))

        # Hide invisible parts
        orth_invisible = zs > 0
        xs[orth_invisible] = np.nan
        ys[orth_invisible] = np.nan
        zs[orth_invisible] = np.nan

        # Hide discontinuities
        if self.lat<0:
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
        self, lat, lst, alt, az, radius=90.0, start_bear=0, end_bear=360, step=1
    ):
        center_ra, center_decl = horizon_to_eq(lat, alt, az, lst, degrees=True)
        eq_circle_points = self.make_circle_points(
            center_ra, center_decl, radius, start_bear, end_bear, step
        )
        ra = np.array(eq_circle_points.data["ra"])
        decl = np.array(eq_circle_points.data["decl"])
        self.lat = lat
        self.lst = lst
        alt, az = self.eq_to_horizon(ra, decl, degrees=True, cart=False)

        circle_data = dict(eq_circle_points.data)
        circle_data["alt"] = alt.tolist()
        circle_data["az"] = az.tolist()

        circle = bokeh.models.ColumnDataSource(data=circle_data)

        return circle

    def make_points(self, points_data):
        points_df = pd.DataFrame(points_data)
        x0s, y0s, z0s = hp.ang2vec(points_df.ra, points_df.decl, lonlat=True).T
        xs, ys, zs = self.to_orth_zenith(x0s, y0s, z0s)

        x_laea, y_laea = self.laea_proj.ang2xy(points_df.ra, points_df.decl, lonlat=True)
        x_moll, y_moll = self.moll_proj.ang2xy(points_df.ra, points_df.decl, lonlat=True)
        x_hz, y_hz = eq_to_horizon(points_df.ra.values, points_df.decl.values, self.lat, self.lst, degrees=True, cart=True)

        # If point_df.ra and points_df.decl have only one value, ang2xy returns scalars (or 0d arrays)
        # not 1d arrays, but bokeh.models.ColumnDataSource requires that column values
        # be python Sequences. So force results of ang2xy to be 1d arrays, even when
        # healpy returns 0d arrays.
        x_laea = x_laea.reshape(x_laea.size)
        y_laea = y_laea.reshape(y_laea.size)
        x_moll = x_moll.reshape(x_moll.size)
        y_moll = y_moll.reshape(y_moll.size)
        x_hz = x_hz.reshape(x_hz.size)
        y_hz = y_hz.reshape(y_hz.size)

        # Hide invisible parts
        invisible = zs > 0
        xs[invisible] = np.nan
        ys[invisible] = np.nan
        zs[invisible] = np.nan
        
        alt, az = eq_to_horizon(points_df.ra, points_df.decl, self.lat, self.lst, degrees=True, cart=False)
        invisible = alt < 0
        x_hz[invisible] = np.nan
        y_hz[invisible] = np.nan

        points = bokeh.models.ColumnDataSource(
            data={
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
        )

        return points


    def add_stars(self, points, plot, lat, sliders=None, star_kwargs={"color": "black"}):
        plot.star(x="x_orth", y="y_orth", size="glyph_size", source=points, **star_kwargs)

        if sliders is not None:
            orth_update_func = bokeh.models.CustomJS(
                args=dict(
                    data_source=points,
                    center_alt_slider=sliders.alt,
                    center_az_slider=sliders.az,
                    lst_slider=sliders.lst,
                    lat=lat,
                ),
                code=self.update_js,
            )

            for slider in sliders:
                slider.js_on_change("value", orth_update_func)

        return plot

    def add_sliders(self):
        self.sliders = OrderedDict()

    def add_lst_slider(self):
        self.sliders["lst"] = bokeh.models.Slider(
            start=-12, end=36, value=self.lst*24/360, step=np.pi / 180, title="LST"
        )

        self.figure = bokeh.layouts.column(
            self.plot, *self.sliders.values()
        )

    def set_js_update_func(self, data_source):
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                center_alt_slider={'value': 90},
                center_az_slider={'value': 0},
                lst_slider=self.sliders['lst'],
                lat=self.lat,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass

    def show(self):
        bokeh.io.show(self.figure)

    def add_healpix(self, data, cmap=None, nside=16, bound_step=1):
        if isinstance(data, bokeh.models.DataSource):
            data_source = data
        else:
            data_source = self.make_healpix_data_source(data, nside, bound_step)
    
        self.healpix_data = data_source
        
        if cmap is None:
            cmap = bokeh.transform.linear_cmap(
                "value", "Inferno256", np.nanmin(data_source.data['value']), np.nanmax(data_source.data['value'])
            )            

        self.healpix_cmap = cmap

        hpgr = self.plot.patches(
            xs=self.x_col,
            ys=self.y_col,
            fill_color=cmap,
            line_color=cmap,
            source=data_source,
        )

        self.healpix_glyph = hpgr.glyph

        hp_glyph = hpgr.glyph

        return data_source, cmap, hp_glyph

    def add_graticules(self, graticule_kwargs={}, line_kwargs={}):
        graticule_points = self.make_graticule_points(**graticule_kwargs)
        kwargs = deepcopy(self.default_graticule_line_kwargs)
        kwargs.update(line_kwargs)
        self.plot.line(
            x=self.x_col, y=self.y_col, source=graticule_points, **kwargs
        )
        return graticule_points

    def add_circle(self, center_ra, center_decl, circle_kwargs={}, line_kwargs={}):
        circle_points = self.make_circle_points(center_ra, center_decl, **circle_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=circle_points, **line_kwargs)
        return circle_points

    def add_horizon(self, zd=90, data_source=None, circle_kwargs={}, line_kwargs={}):
        if data_source is None:
            circle_points = self.make_horizon_circle_points(
                self.lat, self.lst, 90, 0, radius=zd, **circle_kwargs
            )
            if 'lst' in self.sliders:
                self.set_js_update_func(circle_points)
        else:
            circle_points = data_source

        kwargs = deepcopy(self.default_horizon_line_kwargs)
        kwargs.update(line_kwargs)
        self.plot.line(x=self.x_col, y=self.y_col, source=circle_points, **kwargs)
        return circle_points

    def add_marker(self, ra=None, decl=None, name='anonymous', glyph_size=5, data_source=None, circle_kwargs={}):        
        if data_source is None:
            ras = ra if isinstance(ra, Iterable) else [ra]
            decls = decl if isinstance(decl, Iterable) else [decl]
            glyph_sizes = glyph_size if isinstance(glyph_size, Iterable) else [glyph_size]
            names = [name] if isinstance(name, str) else name
            data_source = self.make_points({'ra': ras, 'decl': decls, 'name': names, 'glyph_size': glyph_sizes})
    
        self.plot.circle(
            x=self.x_col,
            y=self.y_col,
            size="glyph_size",
            source=data_source,
            **circle_kwargs,
        )

        return data_source

    def add_stars(self, points_data, data_source=None, mag_limit_slider=False, star_kwargs={}):        
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

            self.sliders['mag_limit'] = mag_slider

        self.figure = bokeh.layouts.column(self.plot, *self.sliders.values())

        return self.star_data_source

    def limit_stars(self, attr, old_limit, mag_limit):
        star_data = self.star_data.query(f"Vmag < {mag_limit}").copy()
        star_data.loc[:, "glyph_size"] = (
            self.max_star_glyph_size
            - (self.max_star_glyph_size / mag_limit) * star_data["Vmag"]
        )
        stars = self.make_points(star_data)
        self.star_data_source.data = dict(stars.data)

    def add_ecliptic(self, **kwargs):
        ecliptic_pole = SkyCoord(lon=0 * u.degree, lat=90 * u.degree, frame="geocentricmeanecliptic").icrs
        line_kwargs = deepcopy(self.default_ecliptic_line_kwargs)
        line_kwargs.update(kwargs)
        points = self.add_circle(ecliptic_pole.ra.deg, ecliptic_pole.dec.deg, line_kwargs=line_kwargs)
        return points
    
    def add_galactic_plane(self, **kwargs):
        galactic_pole = SkyCoord(l=0 * u.degree, b=90 * u.degree, frame="galactic").icrs
        line_kwargs = deepcopy(self.default_galactic_plane_line_kwargs)
        line_kwargs.update(kwargs)
        points = self.add_circle(galactic_pole.ra.deg, galactic_pole.dec.deg, line_kwargs=line_kwargs)
        return points    
    
    def decorate(self, max_zd=70):
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
        data_source, cmap, hp_glyph = super().add_healpix(data, cmap, nside, bound_step)
        self.set_js_update_func(data_source)
        return data_source, cmap, hp_glyph
    
    def add_graticules(self, graticule_kwargs={}, line_kwargs={}):
        data_source = super().add_graticules(graticule_kwargs, line_kwargs)
        self.set_js_update_func(data_source)
        return data_source
    
    def add_circle(self, center_ra, center_decl, circle_kwargs={}, line_kwargs={}):
        data_source = super().add_circle(center_ra, center_decl, circle_kwargs, line_kwargs)
        self.set_js_update_func(data_source)
        return data_source
    
    def add_stars(self, points_data, data_source=None, mag_limit_slider=False, star_kwargs={}):
        data_source = super().add_stars(points_data, data_source, mag_limit_slider, star_kwargs)
        self.set_js_update_func(data_source)
        return data_source        
        
    def add_marker(self, ra=None, decl=None, name='anonymous', glyph_size=5, data_source=None, circle_kwargs={}):
        data_source = super().add_marker(ra, decl, name, glyph_size, data_source, circle_kwargs)
        self.set_js_update_func(data_source)
        return data_source     

class HorizonMap(MovingSphereMap):
    x_col = "x_hz"
    y_col = "y_hz"
    proj_slider_keys = ['lst']
    default_title = "Horizon"

    def set_js_update_func(self, data_source):
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                lst_slider=self.sliders['lst'],
                lat=self.lat,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass

    def add_sliders(self, center_alt=90, center_az=0):
        super().add_sliders()
        self.sliders["lst"] = bokeh.models.Slider(
            start=-12, end=36, value=self.lst * 24/360, step=np.pi / 180, title="LST"
        )

        self.figure = bokeh.layouts.column(
            self.plot, self.sliders['lst']
        )


class ArmillarySphere(MovingSphereMap):
    x_col = "x_orth"
    y_col = "y_orth"
    proj_slider_keys = ['alt', 'az', 'lst']
    default_title = "Armillary Sphere"

    def set_js_update_func(self, data_source):
        update_func = bokeh.models.CustomJS(
            args=dict(
                data_source=data_source,
                center_alt_slider=self.sliders['alt'],
                center_az_slider=self.sliders['az'],
                lst_slider=self.sliders['lst'],
                lat=self.lat,
            ),
            code=self.update_js,
        )

        for proj_slider_key in self.proj_slider_keys:
            try:
                self.sliders[proj_slider_key].js_on_change("value", update_func)
            except KeyError:
                pass


    def add_sliders(self, center_alt=90, center_az=0):
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
        self.sliders["lst"] = bokeh.models.Slider(
            start=-12, end=36, value=self.lst * 24/360, step=np.pi / 180, title="LST"
        )

        self.figure = bokeh.layouts.column(
            self.plot, self.sliders['alt'], self.sliders['az'], self.sliders['lst']
        )

