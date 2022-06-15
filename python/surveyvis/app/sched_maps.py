import bokeh.plotting
from copy import deepcopy
import numpy as np
import healpy as hp
from astropy.time import Time
import logging

import pandas as pd
import bokeh.models

from surveyvis.plot.SphereMap import (
    ArmillarySphere,
    HorizonMap,
    Planisphere,
    MollweideMap,
    make_zscale_linear_cmap,
)

from surveyvis.collect import read_scheduler, read_conditions


def make_logger():
    logger = logging.getLogger('sched_logger')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger

LOGGER = make_logger()


class SchedulerMap():
    tooltips = [
        ("RA", "@center_ra"),
        ("Decl", "@center_decl"),
        ("AvoidDirectWind", "@AvoidDirectWind"),
        ("Moon avoidance", "@Moon_avoidance"),
        ("Zenith shadow mask", "@Zenith_shadow_mask"),
    ]

    def __init__(self, init_key="AvoidDirectWind", nside=16):
        self.scheduler = None
        self.conditions = None
        self.survey_index = [None, None]
        self.scheduler_healpix_maps = {}
        self.init_key = init_key
        self.map_key = init_key
        self.mjd = Time.now().mjd
        self.nside = nside
        self.healpix_cmap = None
        self.data_sources = {}
        self.glyphs = {}
        self.bokeh_models = {}
        self.sphere_maps = {}

    @property
    def healpix_values(self):
        if len(self.scheduler_healpix_maps) == 0:
            npix = hp.nside2npix(self.nside)
            values = np.ones(npix)
            return values

        return self.scheduler_healpix_maps[self.map_key]

    def make_pickle_entry_box(self):
        file_input_box = bokeh.models.TextInput(value='/media/psf/Home/devel/surveyvis/data/Scheduler:2_Scheduler:2_2022-02-18T04:55:02.699.p ', title="Pickle URL:")
        
        def switch_pickle(attrname, old, new):
            LOGGER.info(f"Loading {new}.")
            try:
                self.load(new)
            except FileNotFoundError:
                LOGGER.info("File not found.")
                pass

            LOGGER.debug(f"Finished loading {new}")

        file_input_box.on_change("value", switch_pickle)
        self.bokeh_models['file_input_box'] = file_input_box

    def load(self, file_name):
        self.scheduler = read_scheduler(file_name)
        self.conditions = read_conditions(file_name)

        # FIXME The pickle used for testing does not include several
        # required methods of the Scheduler class, so add them.
        if 'get_basis_functions' not in dir(self.scheduler):
            import surveyvis.munge.monkeypatch_rubin_sim

        self.survey_index[0] = self.scheduler.survey_index[0]
        self.survey_index[1] = self.scheduler.survey_index[1]

        if self.survey_index[0] is None:
            self.survey_index = [0, 0]
        if self.survey_index[1] is None:
            self.survey_index[1] = 0

        self.scheduler_healpix_maps = self.scheduler.get_healpix_maps(
            survey_index=self.survey_index, conditions=self.conditions
        )
        self.map_keys = list(self.scheduler_healpix_maps.keys())
        self.map_key = self.map_keys[-1]

        self.mjd = self.conditions.mjd
        for sphere_map in self.sphere_maps.values():
            sphere_map.mjd = self.mjd

        self.sphere_maps['armillary_sphere'].sliders['lst'].value = self.sphere_maps['armillary_sphere'].lst * 24.0/360.0
        
        self.update_tier_selector()

    def make_tier_selector(self):
        tier_selector = bokeh.models.Select(value=None, options=[None])

        def switch_tier(attrname, old, new):
            self.select_tier(new)

        tier_selector.on_change("value", switch_tier)
        self.bokeh_models['tier_selector'] = tier_selector
        
    def update_tier_selector(self):
        options = [f"tier {t}" for t in np.arange(len(self.scheduler.survey_lists))]
        self.bokeh_models['tier_selector'].options = options
        self.bokeh_models['tier_selector'].value = options[self.survey_index[0]]
        
    def select_tier(self, tier):
        LOGGER.info(f"swiching tier to {tier}")
        self.survey_index[0] = self.bokeh_models['tier_selector'].options.index(tier)
        self.survey_index[1] = 0
        self.update_survey_selector()

    def make_survey_selector(self):
        survey_selector = bokeh.models.Select(value=None, options=[None])

        def switch_survey(attrname, old, new):
            self.select_survey(new)

        survey_selector.on_change("value", switch_survey)
        self.bokeh_models['survey_selector'] = survey_selector
        
    def update_survey_selector(self):
        options = [s.survey_name for s in self.scheduler.survey_lists[self.survey_index[0]]]
        self.bokeh_models['survey_selector'].options = options
        self.bokeh_models['survey_selector'].value = options[self.survey_index[1]]
        
    def select_survey(self, survey):
        tier = self.survey_index[0]
        surveys_in_tier = [s.survey_name for s in self.scheduler.survey_lists[tier]]
        self.survey_index[1] = surveys_in_tier.index(survey)
        
        # Be user we keep using teh same survey_index list, and just update it,
        # not create a new one, because any new one we make won't propogate
        # into other callbacks.
        tier = self.survey_index[0]
        surveys_in_tier = [s.survey_name for s in self.scheduler.survey_lists[tier]]
        self.survey_index[1] = surveys_in_tier.index(survey)

        # Be sure we keep using the same dictionary, and just update it,
        # rather than use a new one because any new one we make won't propogate
        # into other callbacks.
        self.scheduler_healpix_maps.clear()
        self.scheduler_healpix_maps.update(
            self.scheduler.get_healpix_maps(survey_index=self.survey_index, conditions=self.conditions)
        )
        self.map_keys = list(self.scheduler_healpix_maps.keys())

        # Note that updating the value selector triggers the 
        # callback, which updates the maps themselves
        self.update_value_selector()
        
        self.update_reward_table()
        
    def make_value_selector(self):
        value_selector = bokeh.models.Select(value=None, options=[None])

        def switch_value(attrname, old, new):
            LOGGER.info(f"Switching value to {new}")
            self.map_key = new
            self.update_healpix_data()

        value_selector.on_change("value", switch_value)
        self.bokeh_models['value_selector'] = value_selector
        
    def update_value_selector(self):
        self.bokeh_models['value_selector'].options = self.map_keys
        if self.map_key in self.map_keys:
            self.bokeh_models['value_selector'].value = self.map_key
        elif self.init_key in self.map_keys:
            self.bokeh_models['value_selector'].value = self.init_key
        else:
            self.bokeh_models['value_selector'].value = self.map_keys[-1]

    def make_sphere_map(self, key, cls, title, plot_width=512, plot_height=512, decorate=True, horizon_graticules=False):
        plot = bokeh.plotting.figure(
            plot_width=plot_width,
            plot_height=plot_height,
            tooltips=self.tooltips,
            match_aspect=True,
            title=title,
        )
        sphere_map = cls(plot=plot, mjd=self.mjd)
        
        if 'healpix' in self.data_sources:
            sphere_map.add_healpix(self.data_sources['healpix'], cmap=self.healpix_cmap, nside=self.nside)
        else:
            sphere_map.add_healpix(self.healpix_values, nside=self.nside)
            self.data_sources['healpix'] = sphere_map.healpix_data
            self.healpix_cmap = sphere_map.healpix_cmap
            
        if 'horizon' in self.data_sources:
            sphere_map.add_horizon(data_source=self.data_sources['horizon'])
        else:
            self.data_sources['horizon'] = sphere_map.add_horizon()
            
        if 'zd70' in self.data_sources:
            sphere_map.add_horizon(zd=70, data_source=self.data_sources['zd70'], line_kwargs={"color": "red", "line_width": 2})
        else:
            self.data_sources['zd70'] = sphere_map.add_horizon(zd=70, line_kwargs={"color": "red", "line_width": 2})

        if horizon_graticules:
            sphere_map.add_horizon_graticules()

        if decorate:
            sphere_map.decorate()
            
        self.bokeh_models[key] = plot
        self.sphere_maps[key] = sphere_map
        
    def update_healpix_data(self):
        self.healpix_cmap = make_zscale_linear_cmap(self.healpix_values)

        # Pick one of your spherical map objects
        # to use to create a new data source.
        sphere_map = ArmillarySphere(mjd=self.conditions.mjd)
        new_ds = sphere_map.make_healpix_data_source(
            self.healpix_values,
            nside=self.nside,
            bound_step=1,
        )
        new_data = dict(new_ds.data)

        for key in self.map_keys:
            new_data[key] = self.scheduler_healpix_maps[key][new_data["hpid"]]

        # Replace the data to be shown
        self.data_sources['healpix'].data = new_data

        for sphere_map in self.sphere_maps.values():
            sphere_map.healpix_glyph.fill_color = self.healpix_cmap
            sphere_map.healpix_glyph.line_color = self.healpix_cmap

    def make_reward_table(self):
        # Bokeh's DataTable doesn't like to expand to accommodate extra rows, so 
        # create a dummy with lots of rows initially.
        df = pd.DataFrame(np.nan, index=range(30), columns=['basis_function', 'feasible', 'basis_reward', 'accum_reward'])
        self.bokeh_models['reward_table'] = bokeh.models.DataTable(
            source=bokeh.models.ColumnDataSource(df),
            columns=[bokeh.models.TableColumn(field=c, title=c) for c in df]
        )
        
    def update_reward_table(self):
        reward_df = self.scheduler.survey_lists[self.survey_index[0]][self.survey_index[1]].make_reward_df(self.conditions)
        self.bokeh_models['reward_table'].source = bokeh.models.ColumnDataSource(reward_df)
        self.bokeh_models['reward_table'].columns = [bokeh.models.TableColumn(field=c, title=c) for c in reward_df]

    def make_figure(self):
        self.make_sphere_map('armillary_sphere', ArmillarySphere, "Armillary Sphere", plot_width=512, plot_height=512, decorate=True)
        self.bokeh_models['alt_slider'] = self.sphere_maps['armillary_sphere'].sliders['alt']
        self.bokeh_models['az_slider'] = self.sphere_maps['armillary_sphere'].sliders['az']
        self.bokeh_models['lst_slider'] = self.sphere_maps['armillary_sphere'].sliders['lst']
        self.make_sphere_map('planisphere', Planisphere, "Planisphere", plot_width=512, plot_height=512, decorate=True)
        self.make_sphere_map('altaz', HorizonMap, "Alt Az", plot_width=512, plot_height=512, decorate=False, horizon_graticules=True)
        self.make_sphere_map('mollweide', MollweideMap, "Mollweide", plot_width=512, plot_height=512, decorate=True)
        
        self.make_reward_table()
        
        self.make_value_selector()
        self.make_survey_selector()
        self.make_tier_selector()      
        self.make_pickle_entry_box()
       
        controls = [
            self.bokeh_models['alt_slider'],
            self.bokeh_models['az_slider'],
            self.bokeh_models['lst_slider'],
            self.bokeh_models['file_input_box'],
            self.bokeh_models['tier_selector'],
            self.bokeh_models['survey_selector'],
            self.bokeh_models['value_selector'],
        ]

        # self.load('/media/psf/Home/devel/surveyvis/data/Scheduler:2_Scheduler:2_2022-02-18T04:55:02.699.p')

        figure = bokeh.layouts.row(
            bokeh.layouts.column(self.bokeh_models['armillary_sphere'], *controls, self.bokeh_models['reward_table']),
            bokeh.layouts.column(self.bokeh_models['altaz'], self.bokeh_models['planisphere'], self.bokeh_models['mollweide'])
        )

        return figure

        

def make_scheduler_map_figure(
    scheduler_pickle_fname=None, init_key="AvoidDirectWind", nside=16
):
    """Create a set of bekeh figures showing sky maps relevant to scheduler behavior.

    Parameters
    ----------
    scheduler_pickle_fname : `str`, optional
        File from which to load the scheduler state. If set to none, look for the file
        name in the ``SCHED_PICKLE`` environment variable. By default None
    init_key : `str`, optional
        Name of the initial map to show, by default 'AvoidDirectWind'
    nside : int, optional
        Healpix nside to use for display, by default 16

    Returns
    -------
    fig : `bokeh.models.layouts.LayoutDOM`
        A bokeh figure that can be displayed in a notebook (e.g. with ``bokeh.io.show``) or used
        to create a bokeh app.
    """
    scheduler_map = SchedulerMap()
    if scheduler_pickle_fname is not None:
        scheduler_map.load(scheduler_pickle_fname)


    # switch_value("value", init_key, init_key)

    figure = scheduler_map.make_figure()

    return figure


def add_scheduler_map_app(doc):
    """Add a scheduler map figure to a bokeh document

    Parameters
    ----------
    doc : `bokeh.document.document.Document`
        The bokeh document to which to add the figure.
    """
    figure = make_scheduler_map_figure()
    doc.add_root(figure)


if __name__.startswith("bokeh_app_"):
    doc = bokeh.plotting.curdoc()
    add_scheduler_map_app(doc)
