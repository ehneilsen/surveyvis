import bokeh.plotting
import numpy as np
from astropy.time import Time

import pandas as pd
import bokeh.models
import bokeh.core.properties

from surveyvis.plot.SphereMap import (
    ArmillarySphere,
    HorizonMap,
    Planisphere,
    MollweideMap,
)

from surveyvis.collect import sample_pickle

from surveyvis.plot.scheduler import SchedulerDisplay
from surveyvis.plot.scheduler import LOGGER, DEFAULT_NSIDE


class SchedulerDisplayApp(SchedulerDisplay):
    def make_pickle_entry_box(self):
        """Make the entry box for a file name from which to load state."""
        file_input_box = bokeh.models.TextInput(
            value=sample_pickle() + " ",
            title="Pickle path:",
        )

        def switch_pickle(attrname, old, new):
            def do_switch_pickle():
                LOGGER.info(f"Loading {new}.")
                try:
                    self.load(new)
                except FileNotFoundError:
                    LOGGER.info("File not found.")
                    pass

                LOGGER.debug(f"Finished loading {new}")

                # If we do not have access to the document, this won't
                # do anything and is unnecessary, but that's okay.
                self.enable_controls()

            if file_input_box.document is None:
                # If we don't have access to the document, we can't disable
                # the controls, so just do it.
                do_switch_pickle()
            else:
                # disable the controls, and ask the document to do the update
                # on the following event look tick.
                self.disable_controls()
                file_input_box.document.add_next_tick_callback(do_switch_pickle)

        file_input_box.on_change("value", switch_pickle)
        self.bokeh_models["file_input_box"] = file_input_box

    def make_time_selector(self):
        """Create the time selector slider bokeh model."""
        time_selector = bokeh.models.Slider(
            title="MJD",
            start=self.mjd - 1,
            end=self.mjd + 1,
            value=self.mjd,
            step=1.0 / 1440,
        )

        def switch_time(attrname, old, new):
            if time_selector.document is None:
                # If we don't have access to the document, we can't disable
                # the controls, so don't try.
                self.mjd = new
            else:
                # To disable controls as the time is being updated, we need to
                # separate the callback so it happens in two event loop ticks:
                # the first tick disables the controls, the next one
                # actually updates the MJD and then re-enables the controls.
                def do_switch_time():
                    self.mjd = new
                    self.enable_controls()

                self.disable_controls()
                time_selector.document.add_next_tick_callback(do_switch_time)

        time_selector.on_change("value_throttled", switch_time)
        self.bokeh_models["time_selector"] = time_selector
        self.update_time_selector()

    def update_time_selector(self):
        """Update the time selector limits and value to match the date."""
        if "time_selector" in self.bokeh_models:
            self.bokeh_models["time_selector"].start = self.conditions.sun_n12_setting
            self.bokeh_models["time_selector"].end = self.conditions.sun_n12_rising
            self.bokeh_models["time_selector"].value = self.conditions.mjd

    def make_time_input_box(self):
        """Create the time entry box bokeh model."""
        time_input_box = bokeh.models.TextInput(title="Date and time (UTC):")
        self.bokeh_models["time_input_box"] = time_input_box
        self.update_time_input_box()

        def switch_time(attrname, old, new):
            new_mjd = pd.to_datetime(new, utc=True).to_julian_date() - 2400000.5

            if time_input_box.document is None:
                # If we don't have access to the document, we can't disable
                # the controls, so don't try.
                self.mjd = new_mjd
            else:
                # To disable controls as the time is being updated, we need to
                # separate the callback so it happens in two event loop ticks:
                # the first tick disables the controls, the next one
                # actually updates the MJD and then re-enables the controls.
                def do_switch_time():
                    self.mjd = new_mjd
                    self.enable_controls()

                self.disable_controls()
                time_input_box.document.add_next_tick_callback(do_switch_time)

        time_input_box.on_change("value", switch_time)

    def update_time_input_box(self):
        """Update the time selector limits and value to match the date."""
        if "time_input_box" in self.bokeh_models:
            iso_time = Time(self.mjd, format="mjd", scale="utc").iso
            self.bokeh_models["time_input_box"].value = iso_time

    def make_tier_selector(self):
        """Create the tier selector bokeh model."""
        tier_selector = bokeh.models.Select(value=None, options=[None])

        def switch_tier(attrname, old, new):
            self.select_tier(new)

        tier_selector.on_change("value", switch_tier)
        self.bokeh_models["tier_selector"] = tier_selector
        self.update_tier_selector()

    def update_tier_selector(self):
        """Update tier selector to represent tiers for the current survey."""
        if "tier_selector" in self.bokeh_models:
            options = [f"tier {t}" for t in np.arange(len(self.scheduler.survey_lists))]
            self.bokeh_models["tier_selector"].options = options
            self.bokeh_models["tier_selector"].value = options[self.survey_index[0]]

    def select_tier(self, tier):
        """Set the tier being displayed."""
        LOGGER.info(f"swiching tier to {tier}")
        self.survey_index[0] = self.bokeh_models["tier_selector"].options.index(tier)
        self.survey_index[1] = 0
        self.update_survey_selector()

    def make_survey_selector(self):
        """Create the survey selector bokeh model."""
        survey_selector = bokeh.models.Select(value=None, options=[None])

        def switch_survey(attrname, old, new):
            self.select_survey(new)

        survey_selector.on_change("value", switch_survey)
        self.bokeh_models["survey_selector"] = survey_selector

    def update_survey_selector(self):
        """Uptade the survey selector to the current scheduler and tier."""
        if "survey_selector" in self.bokeh_models:
            options = [
                s.survey_name for s in self.scheduler.survey_lists[self.survey_index[0]]
            ]
            self.bokeh_models["survey_selector"].options = options
            self.bokeh_models["survey_selector"].value = options[self.survey_index[1]]

    def make_value_selector(self):
        """Create the bokeh model to select which value to show in maps."""
        value_selector = bokeh.models.Select(value=None, options=[None])

        def switch_value(attrname, old, new):
            LOGGER.info(f"Switching value to {new}")
            self.map_key = new
            self.update_map_data()

        value_selector.on_change("value", switch_value)
        self.bokeh_models["value_selector"] = value_selector

    def update_value_selector(self):
        """Update the value selector bokeh model to show available options."""
        if "value_selector" in self.bokeh_models:
            self.bokeh_models["value_selector"].options = self.map_keys
            if self.map_key in self.map_keys:
                self.bokeh_models["value_selector"].value = self.map_key
            elif self.init_key in self.map_keys:
                self.bokeh_models["value_selector"].value = self.init_key
            else:
                self.bokeh_models["value_selector"].value = self.map_keys[-1]

    def update_time_display(self):
        if "time_selector" in self.bokeh_models:
            self.update_time_selector()

        if "time_input_box" in self.bokeh_models:
            self.update_time_input_box()

    def update_survey_index_display(self):
        self.update_tier_selector()

    def disable_controls(self):
        """Disable all controls.

        Intended to be used while plot elements are updating, and the
        control therefore do not do what the user probably intends.
        """
        LOGGER.info("Disabling controls")
        for model in self.bokeh_models.values():
            try:
                model.disabled = True
            except AttributeError:
                pass

    def enable_controls(self):
        """Enable all controls."""
        LOGGER.info("Enabling controls")
        for model in self.bokeh_models.values():
            try:
                model.disabled = False
            except AttributeError:
                pass

    def make_figure(self):
        self.make_sphere_map(
            "armillary_sphere",
            ArmillarySphere,
            "Armillary Sphere",
            plot_width=512,
            plot_height=512,
            decorate=True,
        )
        self.bokeh_models["alt_slider"] = self.sphere_maps["armillary_sphere"].sliders[
            "alt"
        ]
        self.bokeh_models["az_slider"] = self.sphere_maps["armillary_sphere"].sliders[
            "az"
        ]
        self.bokeh_models["lst_slider"] = self.sphere_maps["armillary_sphere"].sliders[
            "lst"
        ]
        self.bokeh_models["lst_slider"].visible = False
        self.make_sphere_map(
            "planisphere",
            Planisphere,
            "Planisphere",
            plot_width=512,
            plot_height=512,
            decorate=True,
        )
        self.make_sphere_map(
            "altaz",
            HorizonMap,
            "Alt Az",
            plot_width=512,
            plot_height=512,
            decorate=False,
            horizon_graticules=True,
        )
        self.make_sphere_map(
            "mollweide",
            MollweideMap,
            "Mollweide",
            plot_width=512,
            plot_height=512,
            decorate=True,
        )

        self.bokeh_models["key"] = bokeh.models.Div(text=self.key_markup)

        self.make_reward_table()
        self.make_chosen_survey()
        self.make_value_selector()
        self.make_survey_selector()
        self.make_tier_selector()
        self.make_pickle_entry_box()
        self.make_time_selector()

        arm_controls = [
            self.bokeh_models["alt_slider"],
            self.bokeh_models["az_slider"],
        ]

        controls = [self.bokeh_models["file_input_box"]]

        if self.observatory is not None:
            self.make_time_input_box()
            controls.append(self.bokeh_models["time_input_box"])
            controls.append(self.bokeh_models["time_selector"])

        controls += [
            self.bokeh_models["lst_slider"],
            self.bokeh_models["tier_selector"],
            self.bokeh_models["survey_selector"],
            self.bokeh_models["value_selector"],
        ]

        figure = bokeh.layouts.row(
            bokeh.layouts.column(
                self.bokeh_models["altaz"],
                *controls,
                self.bokeh_models["chosen_survey"],
                self.bokeh_models["reward_table"],
            ),
            bokeh.layouts.column(
                self.bokeh_models["planisphere"],
                self.bokeh_models["key"],
                self.bokeh_models["mollweide"],
                self.bokeh_models["armillary_sphere"],
                *arm_controls,
            ),
        )

        return figure


def make_scheduler_map_figure(
    scheduler_pickle_fname=None, init_key="AvoidDirectWind", nside=DEFAULT_NSIDE
):
    """Create a set of bekeh figures showing sky maps for scheduler behavior.

    Parameters
    ----------
    scheduler_pickle_fname : `str`, optional
        File from which to load the scheduler state. If set to none, look for
        the file name in the ``SCHED_PICKLE`` environment variable.
        By default None
    init_key : `str`, optional
        Name of the initial map to show, by default 'AvoidDirectWind'
    nside : int, optional
        Healpix nside to use for display, by default 32

    Returns
    -------
    fig : `bokeh.models.layouts.LayoutDOM`
        A bokeh figure that can be displayed in a notebook (e.g. with
        ``bokeh.io.show``) or used to create a bokeh app.
    """
    scheduler_map = SchedulerDisplayApp(nside=nside)
    figure = scheduler_map.make_figure()

    if scheduler_pickle_fname is not None:
        scheduler_map.load(scheduler_pickle_fname)

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
