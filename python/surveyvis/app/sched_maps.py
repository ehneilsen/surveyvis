import bokeh.plotting
from copy import deepcopy
import numpy as np

from surveyvis.plot.SphereMap import (
    ArmillarySphere,
    HorizonMap,
    Planisphere,
    MollweideMap,
)

from surveyvis.collect import read_scheduler, read_conditions
from surveyvis.munge.scheduler import monkeypatch_scheduler


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

    scheduler = read_scheduler(scheduler_pickle_fname)
    conditions = read_conditions(scheduler_pickle_fname)

    # FIXME The pickle used for testing does not include several
    # required methods of the Scheduler class, so add them.
    scheduler = monkeypatch_scheduler(scheduler)

    survey_index = deepcopy(scheduler.survey_index)

    if survey_index[0] is None:
        survey_index = [0, 0]
    if survey_index[1] is None:
        survey_index[1] = 0

    scheduler_healpix_maps = scheduler.get_healpix_maps(
        survey_index=survey_index, conditions=conditions
    )
    map_keys = list(scheduler_healpix_maps.keys())

    healpy_values = scheduler_healpix_maps[init_key]
    lst = conditions.lmst * 360 / 24

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
    arm = ArmillarySphere(plot=arm_plot, lst=lst)
    hp_ds, cmap, arm_hp_glyph = arm.add_healpix(healpy_values, nside=nside)
    hz = arm.add_horizon()
    zd70 = arm.add_horizon(zd=70, line_kwargs={"color": "red", "line_width": 2})
    arm.decorate()

    pla_plot = bokeh.plotting.figure(
        plot_width=512,
        plot_height=512,
        tooltips=tooltips,
        match_aspect=True,
        title="Planisphere",
    )
    pla = Planisphere(plot=pla_plot, lst=lst)
    pla_hp_ds, pla_cmap, pla_hp_glyph = pla.add_healpix(hp_ds, cmap=cmap, nside=nside)
    pla.add_horizon(data_source=hz)
    pla.add_horizon(
        zd=70, data_source=zd70, line_kwargs={"color": "red", "line_width": 2}
    )
    pla.decorate()

    altaz_plot = bokeh.plotting.figure(
        plot_width=512,
        plot_height=512,
        tooltips=tooltips,
        match_aspect=True,
        title="Horizon",
    )
    altaz = HorizonMap(plot=altaz_plot, lst=lst)
    aa_hp_ds, aa_cmap, aa_hp_glyph = altaz.add_healpix(hp_ds, cmap=cmap, nside=nside)
    # altaz.add_horizon()
    altaz.add_horizon(zd=70, line_kwargs={"color": "red", "line_width": 2})
    # altaz.decorate()

    mol_plot = bokeh.plotting.figure(
        plot_width=512, plot_height=256, tooltips=tooltips, match_aspect=True
    )
    mol = MollweideMap(plot=mol_plot, lst=lst)
    mol_hp_ds, mol_cmap, mol_hp_glyph = mol.add_healpix(hp_ds, cmap=cmap, nside=nside)
    mol.add_horizon(data_source=hz)
    mol.add_horizon(
        zd=70, data_source=zd70, line_kwargs={"color": "red", "line_width": 2}
    )
    mol.decorate()

    #
    # Select survey to show
    #

    surveys_in_tier = [s.survey_name for s in scheduler.survey_lists[survey_index[0]]]
    survey_selector = bokeh.models.Select(
        value=surveys_in_tier[survey_index[0]], options=surveys_in_tier
    )

    def switch_survey(attrname, old, new):
        # Be user we keep using teh same survey_index list, and just update it,
        # not create a new one, because any new one we make won't propogate
        # into other callbacks.
        tier = survey_index[0]
        surveys_in_tier = [s.survey_name for s in scheduler.survey_lists[tier]]
        survey_index[1] = surveys_in_tier.index(new)

        # Be sure we keep using the same dictionary, and just update it,
        # rather than use a new one because any new one we make won't propogate
        # into other callbacks.
        scheduler_healpix_maps.clear()
        scheduler_healpix_maps.update(
            scheduler.get_healpix_maps(survey_index=survey_index, conditions=conditions)
        )
        value_selector.value = init_key

    survey_selector.on_change("value", switch_survey)

    tier_selector = bokeh.models.Select(
        value=f"tier 0",
        options=[f"tier {t}" for t in np.arange(len(scheduler.survey_lists))],
    )

    def switch_tier(attrname, old, new):
        survey_index[0] = tier_selector.options.index(new)
        survey_index[1] = 0
        surveys_in_tier = [
            s.survey_name for s in scheduler.survey_lists[survey_index[0]]
        ]
        survey_selector.value = surveys_in_tier[0]
        survey_selector.options = surveys_in_tier

    tier_selector.on_change("value", switch_tier)

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
            new_data[key] = scheduler_healpix_maps[key][new_data["hpid"]]

        # Replace the data to be shown
        hp_ds.data = new_data

        # Rescale the color map for the new data
        new_cmap = bokeh.transform.linear_cmap(
            "value",
            "Inferno256",
            np.nanmin(new_data["value"]),
            np.nanmax(new_data["value"]),
        )
        arm_hp_glyph.fill_color = new_cmap
        arm_hp_glyph.line_color = new_cmap
        pla_hp_glyph.fill_color = new_cmap
        pla_hp_glyph.line_color = new_cmap
        mol_hp_glyph.fill_color = new_cmap
        mol_hp_glyph.line_color = new_cmap

    value_selector.on_change("value", switch_value)

    switch_value("value", init_key, init_key)

    controls = list(arm.sliders.values()) + [
        tier_selector,
        survey_selector,
        value_selector,
    ]

    row1 = bokeh.layouts.row(
        bokeh.layouts.column(mol.plot, *controls),
        arm.plot,
    )
    row2 = bokeh.layouts.row(altaz.plot, pla.plot)
    figure = bokeh.layouts.column(row1, row2)

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
