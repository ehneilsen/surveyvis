import bokeh.plotting
import numpy as np
import healpy as hp
from astropy.time import Time
import logging

import pandas as pd
import bokeh.models
import bokeh.core.properties

from rubin_sim.scheduler.features.conditions import Conditions
from rubin_sim.scheduler.modelObservatory import Model_observatory
import rubin_sim.scheduler.schedulers
import rubin_sim.scheduler.surveys
import rubin_sim.scheduler.basis_functions


from surveyvis.plot.SphereMap import (
    ArmillarySphere,
    HorizonMap,
    Planisphere,
    MollweideMap,
    make_zscale_linear_cmap,
)

from surveyvis.collect import read_scheduler, read_conditions, sample_pickle

DEFAULT_MJD = 60200.2


def make_logger():
    logger = logging.getLogger("sched_logger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = make_logger()


class SchedulerMap:
    tooltips = [
        ("RA", "@center_ra"),
        ("Decl", "@center_decl"),
        ("AvoidDirectWind", "@AvoidDirectWind"),
        ("Moon avoidance", "@Moon_avoidance"),
        ("Zenith shadow mask", "@Zenith_shadow_mask"),
    ]

    def __init__(self, init_key="AvoidDirectWind", nside=16):
        self._scheduler = None
        self.survey_index = [None, None]
        self.scheduler_healpix_maps = {}
        self.init_key = init_key
        self.map_key = init_key
        self.nside = nside
        self.healpix_cmap = None
        self.data_sources = {}
        self.glyphs = {}
        self.bokeh_models = {}
        self.sphere_maps = {}
        mjd = Time.now().mjd if DEFAULT_MJD is None else DEFAULT_MJD
        try:
            self.observatory = Model_observatory(mjd_start=mjd - 1)
        except ValueError:
            self.observatory = None

        default_scheduler = make_default_scheduler(mjd, nside=nside)
        self.scheduler = default_scheduler

    @property
    def mjd(self):
        return self.conditions.mjd

    @mjd.setter
    def mjd(self, value):
        """Update the interface for a new date

        Parameters
        ----------
        value : `float`
            The new MJD
        """

        # Sometimes a loaded pickle will have close to a represented
        # time, but not identical, and we do not want to try to recreate
        # the conditions object if we have loaded it and not changed the
        # time. So, check only that the mjd is close, not that it
        # is identical.
        if np.abs(value - self.mjd) < (1.0 / (24 * 60 * 60)):
            # Nothing needs to be done
            return

        LOGGER.info(f"Creating conditions for mjd {value}")
        try:
            self.observatory.mjd = value
            conditions = self.observatory.return_conditions()
            LOGGER.info("Conditions created")
        except (ValueError, AttributeError):
            # If we do not have the right cache of sky brightness
            # values on disk, we may not be able to instantiate
            # Model_observatory, but we should be able to run
            # it anyway. Fake up a conditions object as well as
            # we can.
            conditions = Conditions(mjd_start=value - 1)
            conditions.mjd = value
            LOGGER.warning("Created dummy conditions.")

        self.conditions = conditions

    @property
    def healpix_values(self):
        """Healpix numpy array for the current map."""
        if len(self.scheduler_healpix_maps) == 0:
            npix = hp.nside2npix(self.nside)
            values = np.ones(npix)
            return values

        return self.scheduler_healpix_maps[self.map_key]

    def make_pickle_entry_box(self):
        """Make the entry box for a file name from which to load state."""
        file_input_box = bokeh.models.TextInput(
            value=sample_pickle() + " ",
            title="Pickle path:",
        )

        def switch_pickle(attrname, old, new):
            LOGGER.info(f"Loading {new}.")
            try:
                self.load(new)
            except FileNotFoundError:
                LOGGER.info("File not found.")
                pass

            LOGGER.debug(f"Finished loading {new}")

        file_input_box.on_change("value", switch_pickle)
        self.bokeh_models["file_input_box"] = file_input_box

    def load(self, file_name):
        """Load scheduler data

        Parameters
        ----------
        file_name : `str`
            The file name from which to load scheduler state.
        """
        scheduler = read_scheduler(file_name)
        conditions = read_conditions(file_name)
        scheduler.update_conditions(conditions)
        self.scheduler = scheduler

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        """Set the scheduler visualized.

        Parameters
        ----------
        scheduler : `rubin_sim.scheduler.schedulers.core_scheduler.Core_scheduler`  # noqa W505
            The new scheduler to visualize
        """
        LOGGER.debug("Setting the scheduler")
        self._scheduler = scheduler

        # FIXME The pickle used for testing does not include several
        # required methods of the Scheduler class, so add them.
        if "get_basis_functions" not in dir(self.scheduler):
            import surveyvis.munge.monkeypatch_rubin_sim  # noqa F401

        self.survey_index[0] = self.scheduler.survey_index[0]
        self.survey_index[1] = self.scheduler.survey_index[1]

        if self.survey_index[0] is None:
            self.survey_index = [0, 0]
        if self.survey_index[1] is None:
            self.survey_index[1] = 0

        self.update_time_selector()
        self.conditions = scheduler.conditions
        self.update_tier_selector()

    @property
    def conditions(self):
        return self.scheduler.conditions

    @conditions.setter
    def conditions(self, conditions):
        """Update the figure to represent changed conditions.

        Parameters
        ----------
        conditions : `rubin_sim.scheduler.features.conditions.Conditions`
            The new conditions.
        """
        LOGGER.info("Updating interface for new conditions")
        self.scheduler.update_conditions(conditions)
        self.update_reward_table()
        self.scheduler_healpix_maps = self.scheduler.get_healpix_maps(
            survey_index=self.survey_index, conditions=self.conditions
        )

        # If the current map is no longer valid, pick a valid one.
        # Otherwise, keep displaying the same map.
        self.map_keys = list(self.scheduler_healpix_maps.keys())
        if self.map_key not in self.map_keys:
            self.map_key = self.map_keys[-1]

        for sphere_map in self.sphere_maps.values():
            sphere_map.mjd = self.mjd

        if "armillary_sphere" in self.sphere_maps:
            self.sphere_maps["armillary_sphere"].sliders["lst"].value = (
                self.sphere_maps["armillary_sphere"].lst * 24.0 / 360.0
            )

        if "time_selector" in self.bokeh_models:
            self.update_time_selector()

        if "time_input_box" in self.bokeh_models:
            self.update_time_input_box()

        # Actually push the change out to the user's browser
        self.update_healpix_data()

        LOGGER.info("Finished updating conditions")

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
            self.mjd = new

        time_selector.on_change("value_throttled", switch_time)
        self.bokeh_models["time_selector"] = time_selector

    def update_time_selector(self):
        """Update the time selector limits and value to match the date."""
        if "time_selector" in self.bokeh_models:
            self.bokeh_models["time_selector"].start = self.conditions.sunset
            self.bokeh_models["time_selector"].end = self.conditions.sunrise
            self.bokeh_models["time_selector"].value = self.conditions.mjd

    def make_time_input_box(self):
        """Create the time entry box bokeh model."""
        time_input_box = bokeh.models.TextInput(title="Date and time (UTC):")
        self.bokeh_models["time_input_box"] = time_input_box
        self.update_time_input_box()

        def switch_time(attrname, old, new):
            new_mjd = pd.to_datetime(new, utc=True).to_julian_date() - 2400000.5
            LOGGER.debug(
                f"Old mjd: {self.mjd}, New MJD: {new_mjd}, equal: {self.mjd==new_mjd}, diff: {self.mjd-new_mjd}"
            )
            self.mjd = new_mjd

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

    def select_survey(self, survey):
        """Update the display to show a given survey.

        Parameters
        ----------
        survey : `str`
            The name of the survey to select.
        """
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
            self.scheduler.get_healpix_maps(
                survey_index=self.survey_index, conditions=self.conditions
            )
        )
        self.map_keys = list(self.scheduler_healpix_maps.keys())

        # Note that updating the value selector triggers the
        # callback, which updates the maps themselves
        self.update_value_selector()

        self.update_reward_table()

    def make_value_selector(self):
        """Create the bokeh model to select which value to show in maps."""
        value_selector = bokeh.models.Select(value=None, options=[None])

        def switch_value(attrname, old, new):
            LOGGER.info(f"Switching value to {new}")
            self.map_key = new
            self.update_healpix_data()

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

    def make_sphere_map(
        self,
        key,
        cls,
        title,
        plot_width=512,
        plot_height=512,
        decorate=True,
        horizon_graticules=False,
    ):
        plot = bokeh.plotting.figure(
            plot_width=plot_width,
            plot_height=plot_height,
            tooltips=self.tooltips,
            match_aspect=True,
            title=title,
        )
        sphere_map = cls(plot=plot, mjd=self.mjd)

        if "healpix" in self.data_sources:
            sphere_map.add_healpix(
                self.data_sources["healpix"], cmap=self.healpix_cmap, nside=self.nside
            )
        else:
            sphere_map.add_healpix(self.healpix_values, nside=self.nside)
            self.data_sources["healpix"] = sphere_map.healpix_data
            self.healpix_cmap = sphere_map.healpix_cmap

        if "horizon" in self.data_sources:
            sphere_map.add_horizon(data_source=self.data_sources["horizon"])
        else:
            self.data_sources["horizon"] = sphere_map.add_horizon()

        if "zd70" in self.data_sources:
            sphere_map.add_horizon(
                zd=70,
                data_source=self.data_sources["zd70"],
                line_kwargs={"color": "red", "line_width": 2},
            )
        else:
            self.data_sources["zd70"] = sphere_map.add_horizon(
                zd=70, line_kwargs={"color": "red", "line_width": 2}
            )

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
        if "healpix" in self.data_sources:
            self.data_sources["healpix"].data = new_data

        for sphere_map in self.sphere_maps.values():
            sphere_map.healpix_glyph.fill_color = self.healpix_cmap
            sphere_map.healpix_glyph.line_color = self.healpix_cmap

    def make_reward_table(self):
        # Bokeh's DataTable doesn't like to expand to accommodate extra rows,
        # so create a dummy with lots of rows initially.
        df = pd.DataFrame(
            np.nan,
            index=range(30),
            columns=["basis_function", "feasible", "basis_reward", "accum_reward"],
        )
        self.bokeh_models["reward_table"] = bokeh.models.DataTable(
            source=bokeh.models.ColumnDataSource(df),
            columns=[bokeh.models.TableColumn(field=c, title=c) for c in df],
        )

    def update_reward_table(self):
        if "reward_table" in self.bokeh_models:
            reward_df = self.scheduler.survey_lists[self.survey_index[0]][
                self.survey_index[1]
            ].make_reward_df(self.conditions)
            self.bokeh_models["reward_table"].source = bokeh.models.ColumnDataSource(
                reward_df
            )
            self.bokeh_models["reward_table"].columns = [
                bokeh.models.TableColumn(field=c, title=c) for c in reward_df
            ]

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

        self.make_reward_table()

        self.make_value_selector()
        self.make_survey_selector()
        self.make_tier_selector()
        self.make_pickle_entry_box()
        self.make_time_selector()

        controls = [
            self.bokeh_models["alt_slider"],
            self.bokeh_models["az_slider"],
        ]

        if self.observatory is not None:
            self.make_time_input_box()
            controls.append(self.bokeh_models["time_input_box"])
            controls.append(self.bokeh_models["time_selector"])

        controls += [
            self.bokeh_models["lst_slider"],
            self.bokeh_models["file_input_box"],
            self.bokeh_models["tier_selector"],
            self.bokeh_models["survey_selector"],
            self.bokeh_models["value_selector"],
        ]

        figure = bokeh.layouts.row(
            bokeh.layouts.column(
                self.bokeh_models["armillary_sphere"],
                *controls,
                self.bokeh_models["reward_table"],
            ),
            bokeh.layouts.column(
                self.bokeh_models["altaz"],
                self.bokeh_models["planisphere"],
                self.bokeh_models["mollweide"],
            ),
        )

        return figure


def make_scheduler_map_figure(
    scheduler_pickle_fname=None, init_key="AvoidDirectWind", nside=16
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
        Healpix nside to use for display, by default 16

    Returns
    -------
    fig : `bokeh.models.layouts.LayoutDOM`
        A bokeh figure that can be displayed in a notebook (e.g. with
        ``bokeh.io.show``) or used to create a bokeh app.
    """
    scheduler_map = SchedulerMap()
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


def make_default_scheduler(mjd, nside=32):
    """Return default scheduler.

    Parameters
    ----------
    mjd : `float`
        The MJD.
    nside : `int`
        The healpix nside

    Returns
    -------
    scheduler : `rubin_sim.scheduler.schedulers.Core_scheduler`
    """
    LOGGER.debug("Making default scheduler")

    def make_band_survey(band):
        # Split the creation of basis functions so that if one fails,
        # the other(s) might still be included.
        basis_functions = []
        try:
            this_basis_function = (
                rubin_sim.scheduler.basis_functions.Ecliptic_basis_function(nside=nside)
            )
            basis_functions.append(this_basis_function)
        except Exception:
            pass

        try:
            this_basis_function = (
                rubin_sim.scheduler.basis_functions.M5_diff_basis_function(
                    filtername=band, nside=nside
                )
            )
            basis_functions.append(this_basis_function)
        except Exception:
            pass

        survey = rubin_sim.scheduler.surveys.BaseSurvey(
            basis_functions,
            survey_name=band,
        )
        return survey

    band_surveys = {b: make_band_survey(b) for b in "ugrizy"}
    visible_surveys = [band_surveys["u"], band_surveys["g"], band_surveys["r"]]
    ir_surveys = [band_surveys["i"], band_surveys["z"], band_surveys["y"]]

    scheduler = rubin_sim.scheduler.schedulers.Core_scheduler(
        [visible_surveys, ir_surveys], nside=nside
    )
    try:
        observatory = Model_observatory(mjd_start=mjd - 1)
        observatory.mjd = mjd
        conditions = observatory.return_conditions()
    except ValueError:
        # If we do not have the right cache of sky brightness
        # values on disk, we may not be able to instantiate
        # Model_observatory, but we should be able to run
        # it anyway. Fake up a conditions object as well as
        # we can.
        conditions = Conditions(mjd_start=mjd - 1)
        conditions.mjd = mjd

    scheduler.update_conditions(conditions)
    return scheduler


if __name__.startswith("bokeh_app_"):
    doc = bokeh.plotting.curdoc()
    add_scheduler_map_app(doc)
