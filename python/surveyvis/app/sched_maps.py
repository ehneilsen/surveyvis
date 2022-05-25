import os
import bokeh.plotting
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
import healpy as hp

from rubin_sim import maf

from surveyvis.plot.SphereMap import (
    ArmillarySphere,
    Planisphere,
    MollweideMap
)

from surveyvis.collect.SchedulerState import SchedulerState

def make_metric_figure(scheduler_pickle_fname=None, init_key='AvoidDirectWind', nside=16):
    if scheduler_pickle_fname is None:
        scheduler_pickle_fname = os.environ['SCHED_PICKLE']

    scheduler_state = SchedulerState(scheduler_pickle_fname)
    scheduler_healpix_maps = scheduler_state.healpix_map
    map_keys = list(scheduler_healpix_maps.keys())

    healpy_values = scheduler_healpix_maps[init_key]
    
    tooltips = [
        ("RA", "@center_ra"),
        ("Decl", "@center_decl"),
        ("AvoidDirectWind", "@AvoidDirectWind"),
        ("Slewtime", "@Slewtime_basis_function"),
        ("Moon avoidance", "@Moon_avoidance_basis_function"),
        ("Zenith shadow mask", "@Zenith_shadow_mask_basis_function"),
    ]

    arm_plot = bokeh.plotting.figure(
        plot_width=512,
        plot_height=512,
        tooltips=tooltips,
        match_aspect=True,
        title="Armillary sphere",
    )
    arm = ArmillarySphere(plot=arm_plot)
    hp_ds, cmap, arm_hp_glyph = arm.add_healpix(healpy_values, nside=nside)
    hz = arm.add_horizon()
    zd70 = arm.add_horizon(zd=70, line_kwargs={"color": "red", "line_width": 2})
    arm.decorate()
    
    pla_plot = bokeh.plotting.figure(
        plot_width=512,
        plot_height=512,
        tooltips=tooltips,
        match_aspect=True,
        title="Planisphere sphere",
    )
    pla = Planisphere(plot=pla_plot)
    pla_hp_ds, pla_cmap, pla_hp_glyph = pla.add_healpix(hp_ds, cmap=cmap, nside=nside)
    pla.add_horizon(data_source=hz)
    pla.add_horizon(zd=70, data_source=zd70, line_kwargs={"color": "red", "line_width": 2})
    pla.decorate()
    
    mol_plot = bokeh.plotting.figure(
                plot_width=512, plot_height=256, tooltips=tooltips, match_aspect=True
            )
    mol = MollweideMap(plot=mol_plot)
    mol_hp_ds, mol_cmap, mol_hp_glyph = mol.add_healpix(hp_ds, cmap=cmap, nside=nside)
    mol.add_horizon(data_source=hz)
    mol.add_horizon(zd=70, data_source=zd70, line_kwargs={"color": "red", "line_width": 2})
    mol.decorate()
    
    #
    # Select scheduler to show
    # 
    
    surveys_in_tier = [s.survey_name for s in scheduler_state.sched.survey_lists[0]]
    survey_selector = bokeh.models.Select(
        value=surveys_in_tier[0],
        options=surveys_in_tier
    )
    
    def switch_survey(attrname, old, new):
        tier = scheduler_state.survey_list_indexes[0]
        surveys_in_tier = [s.survey_name for s in scheduler_state.sched.survey_lists[tier]]
        scheduler_state.survey_list_indexes = (tier, surveys_in_tier.index(new))
        switch_value(None, None, init_key)
    
    survey_selector.on_change('value', switch_survey)
    
    tier_selector = bokeh.models.Select(
        value=f'tier 0',
        options=[f'tier {t}' for t in np.arange(len(scheduler_state.sched.survey_lists))]
    )
    
    def switch_tier(attrname, old, new):
        new_tier_index = tier_selector.options.index(new)
        scheduler_state.survey_list_indexes = (new_tier_index, 0)
        surveys_in_tier = [s.survey_name for s in scheduler_state.sched.survey_lists[new_tier_index]]
        survey_selector.value = surveys_in_tier[0]
        survey_selector.options = surveys_in_tier
        switch_value(None, None, init_key)
        
    tier_selector.on_change('value', switch_tier)
    
    #
    # Select which map to show
    #
    
    value_selector = bokeh.models.Select(
        value=init_key,
        options=map_keys,
    )

    def switch_value(attrname, old, new):
        hp_values = scheduler_healpix_maps[new]
            
        new_ds = arm.make_healpix_data_source(
            hp_values,
            nside=nside,
            bound_step=1,
        )
        new_data = dict(new_ds.data)
        
        for key in map_keys:
            new_data[key] = scheduler_healpix_maps[key][new_data['hpid']]

        # Replace the data to be shown
        hp_ds.data = new_data
        
        # Rescale the color map for the new data
        new_cmap = bokeh.transform.linear_cmap("value", "Inferno256", np.nanmin(new_data['value']), np.nanmax(new_data['value']))
        arm_hp_glyph.fill_color = new_cmap   
        arm_hp_glyph.line_color = new_cmap
        pla_hp_glyph.fill_color = new_cmap   
        pla_hp_glyph.line_color = new_cmap
        mol_hp_glyph.fill_color = new_cmap   
        mol_hp_glyph.line_color = new_cmap

    value_selector.on_change("value", switch_value)
    
    switch_value('value', init_key, init_key)

    controls = list(arm.sliders.values()) + [tier_selector, survey_selector, value_selector]
    figure = bokeh.layouts.row(
        bokeh.layouts.column(mol.plot, *controls),
        arm.plot, 
        pla.plot
    )
    
    return figure

def add_metric_app(doc):
    figure = make_metric_figure()
    doc.add_root(figure)

if __name__.startswith('bokeh_app_'):
    doc = bokeh.plotting.curdoc()
    add_metric_app(doc)

