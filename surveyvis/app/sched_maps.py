import bokeh.plotting
import numpy as np
import healpy as hp
from astropy.time import Time
import logging
import collections.abc
from collections import OrderedDict

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

from surveyvis.collect import read_scheduler, sample_pickle

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
    ]

    key_markup = """<h1>Key</h1>
<ul>
<li><b>Black line</b> Horizon</li>
<li><b>Red line</b> ZD=70 deg.</li>
<li><b>Green line</b> Ecliptic</li>
<li><b>Blue line</b> Galactic plane</li>
<li><b>Yellow dot</b> Sun position</li>
<li><b>Gray dot</b> Moon position</li>
<li><b>Red dot</b> Survey field(s)</li>
<li><b>Greed dot</b> Telescope pointing</li>
</ul>
    """

    def __init__(self, init_key="AvoidDirectWind", nside=16):
        self._scheduler = None
        self.survey_index = [None, None]
        self.scheduler_healpix_maps = OrderedDict()
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
    def map_keys(self):
        """Return keys for the available healpix maps"""
        keys = list(self.scheduler_healpix_maps.keys())
        return keys

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

    def _update_scheduler_healpix_maps(self):
        """Update healpix values from the scheduler."""
        # Be sure we keep using the same dictionary, and just update it,
        # rather than use a new one because any new one we make won't propogate
        # into other callbacks.
        self.scheduler_healpix_maps.clear()
        full_healpix_maps = self.scheduler.get_healpix_maps(
            survey_index=self.survey_index, conditions=self.conditions
        )
        for key in full_healpix_maps:
            new_key = key.replace(" ", "_").replace(".", "_")
            self.scheduler_healpix_maps[new_key] = hp.ud_grade(
                full_healpix_maps[key], self.nside
            )

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

    def load(self, file_name):
        """Load scheduler data

        Parameters
        ----------
        file_name : `str`
            The file name from which to load scheduler state.
        """
        scheduler, conditions = read_scheduler(file_name)
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
        self.scheduler.request_observation()
        self.update_chosen_survey()
        self.update_reward_table()
        self._update_scheduler_healpix_maps()

        # If the current map is no longer valid, pick a valid one.
        # Otherwise, keep displaying the same map.
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
        self.update_map_data()

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

        # Be user we keep using the same survey_index list, and just update it,
        # not create a new one, because any new one we make won't propogate
        # into other callbacks.
        tier = self.survey_index[0]
        surveys_in_tier = [s.survey_name for s in self.scheduler.survey_lists[tier]]
        self.survey_index[1] = surveys_in_tier.index(survey)
        self._update_scheduler_healpix_maps()

        # Note that updating the value selector triggers the
        # callback, which updates the maps themselves
        self.update_value_selector()
        self.update_survey_marker_data()
        self.update_reward_table()
        self.update_hovertool()

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

        if "hover_tool" not in self.bokeh_models:
            self.bokeh_models["hover_tool"] = bokeh.models.HoverTool(
                renderers=[], tooltips=self.tooltips
            )

        plot = bokeh.plotting.figure(
            plot_width=plot_width,
            plot_height=plot_height,
            tools=[self.bokeh_models["hover_tool"]],
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

        self.bokeh_models["hover_tool"].renderers.append(sphere_map.healpix_renderer)

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

        if "survey_marker" not in self.data_sources:
            self.data_sources["survey_marker"] = self.make_survey_marker_data_source(
                sphere_map
            )

        sphere_map.add_marker(
            data_source=self.data_sources["survey_marker"],
            name="Field",
            circle_kwargs={"color": "red", "fill_alpha": 0.5},
        )

        if "telescope_marker" not in self.data_sources:
            self.data_sources[
                "telescope_marker"
            ] = self.make_telescope_marker_data_source(sphere_map)

        sphere_map.add_marker(
            data_source=self.data_sources["telescope_marker"],
            name="Field",
            circle_kwargs={"color": "green", "fill_alpha": 0.5},
        )

        if "moon_marker" not in self.data_sources:
            self.data_sources["moon_marker"] = self.make_moon_marker_data_source(
                sphere_map
            )

        sphere_map.add_marker(
            data_source=self.data_sources["moon_marker"],
            name="Moon",
            circle_kwargs={"color": "lightgray", "fill_alpha": 0.8},
        )

        if "sun_marker" not in self.data_sources:
            self.data_sources["sun_marker"] = self.make_moon_marker_data_source(
                sphere_map
            )

        sphere_map.add_marker(
            data_source=self.data_sources["sun_marker"],
            name="Sun",
            circle_kwargs={"color": "yellow", "fill_alpha": 1},
        )

        self.bokeh_models[key] = plot
        self.sphere_maps[key] = sphere_map

    def _make_marker_data_source(
        self,
        sphere_map=None,
        name="telescope",
        source_name="conditions",
        ra_name="telRA",
        decl_name="telDec",
        source_units="radians",
    ):
        """Create a bokeh datasource for the moon.

        Parameters
        ----------
        sphere_map: `surveyvis.plot.SphereMap`
            The instance of SphereMap to use to create the data source
        name : 'str'
            The name of the thing to mark.
        source_name : `str`
            The name of the member object to provide the coordinates.
        ra_name : `str`
            The name of the member with the RA.
        decl_name : `str`
            The name of the member with the declination.
        source_units : `str`
            'radians' or 'degrees', according to what is provided by the source

        Returns
        -------
        data_source: `bokeh.models.ColumnDataSource`
            The DataSource with the column data.
        """
        if sphere_map is None:
            sphere_map = tuple(self.sphere_maps.values())[0]

        sources = {
            "conditions": self.conditions,
            "survey": self.scheduler.survey_lists[self.survey_index[0]][
                self.survey_index[1]
            ],
        }
        source = sources[source_name]

        # If the telescope position is not set in our instance of
        # conditions, use an empty array
        ra = getattr(source, ra_name, np.array([]))
        decl = getattr(source, decl_name, np.array([]))
        if ra is None:
            ra = np.array([])
        if decl is None:
            decl = np.array([])
        LOGGER.debug(
            f"{name} coordinates: ra={np.degrees(ra)}, decl={np.degrees(decl)}"
        )
        if source_units == "radians":
            ra_deg = np.degrees(ra)
            decl_deg = np.degrees(decl)
        elif source_units in ("degrees", "deg"):
            ra_deg = ra
            decl_deg = decl
        data_source = sphere_map.make_marker_data_source(
            ra=ra_deg, decl=decl_deg, name=name, glyph_size=20
        )
        return data_source

    def make_moon_marker_data_source(self, sphere_map=None):
        """Create a bokeh datasource for the moon.

        Parameters
        ----------
        sphere_map: `surveyvis.plot.SphereMap`
            The instance of SphereMap to use to create the data source

        Returns
        -------
        data_source: `bokeh.models.ColumnDataSource`
            The DataSource with the column data.
        """
        data_source = self._make_marker_data_source(
            sphere_map=sphere_map,
            name="moon",
            source_name="conditions",
            ra_name="moonRA",
            decl_name="moonDec",
            source_units="radians",
        )
        return data_source

    def update_moon_marker_data(self):
        """Update the moon data source."""
        if "telescope_marker" not in self.data_sources:
            return

        sphere_map = tuple(self.sphere_maps.values())[0]
        data_source = self.make_moon_marker_data_source(sphere_map)
        data = dict(data_source.data)
        if "moon_marker" in self.data_sources:
            self.data_sources["moon_marker"].data = data

    def make_sun_marker_data_source(self, sphere_map=None):
        """Create a bokeh datasource for the sun.

        Parameters
        ----------
        sphere_map: `surveyvis.plot.SphereMap`
            The instance of SphereMap to use to create the data source

        Returns
        -------
        data_source: `bokeh.models.ColumnDataSource`
            The DataSource with the column data.
        """
        data_source = self._make_marker_data_source(
            sphere_map=sphere_map,
            name="sun",
            source_name="conditions",
            ra_name="sunRA",
            decl_name="sunDec",
            source_units="radians",
        )
        return data_source

    def update_sun_marker_data(self):
        """Update the sun data source."""
        if "telescope_marker" not in self.data_sources:
            return

        sphere_map = tuple(self.sphere_maps.values())[0]
        data_source = self.make_sun_marker_data_source(sphere_map)
        data = dict(data_source.data)
        if "sun_marker" in self.data_sources:
            self.data_sources["sun_marker"].data = data

    def make_telescope_marker_data_source(self, sphere_map=None):
        """Create a bokeh datasource for the current telescope pointing.

        Parameters
        ----------
        sphere_map: `surveyvis.plot.SphereMap`
            The instance of SphereMap to use to create the data source

        Returns
        -------
        data_source: `bokeh.models.ColumnDataSource`
            The DataSource with the column data.
        """
        data_source = self._make_marker_data_source(
            sphere_map=sphere_map,
            name="telescope",
            source_name="conditions",
            ra_name="telRA",
            decl_name="telDec",
            source_units="radians",
        )
        return data_source

    def update_telescope_marker_data(self):
        """Update the telescope pointing data source."""
        if "telescope_marker" not in self.data_sources:
            return

        sphere_map = tuple(self.sphere_maps.values())[0]
        data_source = self.make_telescope_marker_data_source(sphere_map)
        data = dict(data_source.data)
        if "telescope_marker" in self.data_sources:
            self.data_sources["telescope_marker"].data = data

    def make_survey_marker_data_source(self, sphere_map=None):
        """Create a bokeh datasource for the pointings for the current survey.

        Parameters
        ----------
        sphere_map: `surveyvis.plot.SphereMap`
            The instance of SphereMap to use to create the data source

        Returns
        -------
        data_source: `bokeh.models.ColumnDataSource`
            The DataSource with the column data.
        """
        data_source = self._make_marker_data_source(
            sphere_map=sphere_map,
            name="Field",
            source_name="survey",
            ra_name="ra",
            decl_name="dec",
            source_units="radians",
        )
        return data_source

    def update_survey_marker_data(self):
        """Update the survey pointing data source."""
        if "survey_marker" not in self.data_sources:
            return

        sphere_map = tuple(self.sphere_maps.values())[0]
        data_source = self.make_survey_marker_data_source(sphere_map)
        data = dict(data_source.data)
        if "survey_marker" in self.data_sources:
            self.data_sources["survey_marker"].data = data

    def update_healpix_data(self):
        """Update the healpix value data source."""
        if "healpix" not in self.data_sources:
            return

        sphere_map = tuple(self.sphere_maps.values())[0]
        # sphere_map = ArmillarySphere(mjd=self.conditions.mjd)

        if "Zenith_shadow_mask" in self.map_keys:
            zenith_mask = self.scheduler_healpix_maps["Zenith_shadow_mask"]
            cmap_sample_data = self.healpix_values[zenith_mask == 1]
        elif "y_sky" in self.map_keys:
            sb_mask = self.scheduler_healpix_maps["y_sky"] > 10
            cmap_sample_data = self.healpix_values[sb_mask]
            if len(cmap_sample_data) == 0:
                # It's probably day, so the color map will be bad regardless.
                cmap_sample_data = self.healpix_values
        else:
            cmap_sample_data = self.healpix_values

        self.healpix_cmap = make_zscale_linear_cmap(cmap_sample_data)

        new_ds = sphere_map.make_healpix_data_source(
            self.healpix_values,
            nside=self.nside,
            bound_step=1,
        )
        new_data = dict(new_ds.data)

        for key in self.map_keys:
            # The datasource might not have the healpixels in the same order
            # so force the order by indexing on new_data["hpid"]
            new_data[key] = self.scheduler_healpix_maps[key][new_data["hpid"]]

        # Replace the data to be shown
        self.data_sources["healpix"].data = new_data

        for sphere_map in self.sphere_maps.values():
            sphere_map.healpix_glyph.fill_color = self.healpix_cmap
            sphere_map.healpix_glyph.line_color = self.healpix_cmap

    def update_hovertool(self):
        """Update the hovertool with available value."""
        if "hover_tool" not in self.bokeh_models:
            return

        tooltips = []
        data = self.data_sources["healpix"].data
        for data_key in data.keys():
            if not isinstance(data[data_key][0], collections.abc.Sequence):

                if data_key == "center_ra":
                    label = "RA"
                elif data_key == "center_decl":
                    label = "Decl"
                else:
                    label = data_key.replace("_", " ")

                reference = f"@{data_key}"
                tooltips.append((label, reference))

        self.bokeh_models["hover_tool"].tooltips = tooltips

    def update_map_data(self):
        """Update all map related bokeh data sources"""
        self.update_healpix_data()
        self.update_telescope_marker_data()
        self.update_moon_marker_data()
        self.update_sun_marker_data()
        self.update_survey_marker_data()

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

    def make_chosen_survey(self):
        self.bokeh_models["chosen_survey"] = bokeh.models.Div(
            text="<p>No chosen survey</p>"
        )

    def update_chosen_survey(self):
        if "chosen_survey" in self.bokeh_models:
            tier = f"tier {self.scheduler.survey_index[0]}"
            survey = self.scheduler.survey_lists[self.scheduler.survey_index[0]][
                self.scheduler.survey_index[1]
            ].survey_name
            self.bokeh_models[
                "chosen_survey"
            ].text = f"<p>Chosen survey: {tier}, {survey}</p>"

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
                self.bokeh_models["key"],
                self.bokeh_models["armillary_sphere"],
                *controls,
                self.bokeh_models["chosen_survey"],
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
    scheduler.request_observation()
    return scheduler


if __name__.startswith("bokeh_app_"):
    doc = bokeh.plotting.curdoc()
    add_scheduler_map_app(doc)
