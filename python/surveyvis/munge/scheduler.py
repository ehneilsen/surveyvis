from io import StringIO
from collections import OrderedDict
import rubin_sim


def monkeypatch_scheduler(scheduler):
    """Add methods to a scheduler object to support display.

    Parameters
    ----------
    scheduler : `rubin_sim.scheduler.schedulers.core_scheduler.Core_scheduler`
        An instance of the scheduler to patch.

    Returns
    -------
    scheduler : `rubin_sim.scheduler.schedulers.core_scheduler.Core_scheduler`
        The patched instance of the scheduler. (The same instance provided as an
        argument.)
    """

    # Patching for Core_scheduler itself

    def get_basis_functions(self, survey_index=None, conditions=None):
        """Get the basis functions for a specific survey, in provided conditions.

        Parameters
        ----------
        survey_index : `List` [`int`], optional
            A list with two elements: the survey list and the element within that
            survey list for which the basis function should be retrieved. If ``None``,
            use the latest survey to make an addition to the queue.
        conditions : `rubin_sim.scheduler.features.conditions.Conditions`, optional
            The conditions for which to return the basis functions. If ``None``, use
            the conditions associated with this sceduler. By default None.

        Returns
        -------
        basis_funcs : `OrderedDict` ['str`, `rubin_sim.scheduler.basis_functions.basis_functions.Base_basis_function`]
            A dictionary of the basis functions, where the keys are names for the basis functions and the values
            are the functions themselves.
        """
        if survey_index is None:
            survey_index = self.survey_index

        if conditions is None:
            conditions = self.conditions

        survey = self.survey_lists[survey_index[0]][survey_index[1]]
        basis_funcs = OrderedDict()
        for basis_func in survey.basis_functions:
            if hasattr(basis_func(conditions), "__len__"):
                basis_funcs[basis_func.__class__.__name__] = basis_func
        return basis_funcs
    
    if "get_basis_functions" not in dir(scheduler):
        scheduler.__class__.get_basis_functions = get_basis_functions

    def get_healpix_maps(self, survey_index=None, conditions=None):
        """Get the healpix maps for a specific survey, in provided conditions.

        Parameters
        ----------
        survey_index : `List` [`int`], optional
            A list with two elements: the survey list and the element within that
            survey list for which the maps that should be retrieved. If ``None``,
            use the latest survey to make an addition to the queue.
        conditions : `rubin_sim.scheduler.features.conditions.Conditions`, optional
            The conditions for the maps to be returned. If ``None``, use
            the conditions associated with this sceduler. By default None.

        Returns
        -------
        basis_funcs : `OrderedDict` ['str`, `numpy.ndarray`]
            A dictionary of the maps, where the keys are names for the maps and
            values are the numpy arrays as used by ``healpy``.
        """
        if survey_index is None:
            survey_index = self.survey_index

        if conditions is None:
            conditions = self.conditions

        maps = OrderedDict()
        for band in conditions.skybrightness.keys():
            maps[f"{band}_sky"] = conditions.skybrightness[band]

        basis_functions = self.get_basis_functions(survey_index, conditions)

        for basis_func_key in basis_functions.keys():
            maps[basis_func_key] = basis_functions[basis_func_key](conditions)

        return maps
    
    if "get_healpix_maps" not in dir(scheduler):
        scheduler.__class__.get_healpix_maps = get_healpix_maps

    def scheduler_repr(self):
        if isinstance(
            self.pointing2hpindx, rubin_sim.scheduler.utils.utils.hp_in_lsst_fov
        ):
            camera = "LSST"
        elif isinstance(
            self.pointing2hpindx, rubin_sim.scheduler.utils.utils.hp_in_comcam_fov
        ):
            camera = "comcam"
        else:
            camera = None

        this_repr = f"""{self.__class__.__qualname__}(
            surveys={repr(self.survey_lists)},
            camera="{camera}",
            nside={repr(self.nside)},
            rotator_limits={repr(self.rotator_limits)},
            survey_index={repr(self.survey_index)},
            log={repr(self.log)}
        )"""
        return this_repr

    scheduler.__class__.__repr__ = scheduler_repr

    def scheduler_str(self):
        if isinstance(
            self.pointing2hpindx, rubin_sim.scheduler.utils.utils.hp_in_lsst_fov
        ):
            camera = "LSST"
        elif isinstance(
            self.pointing2hpindx, rubin_sim.scheduler.utils.utils.hp_in_comcam_fov
        ):
            camera = "comcam"
        else:
            camera = None

        output = StringIO()
        print(self.__class__.__qualname__, file=output)
        if len(self.survey_lists) == 0:
            print("Scheduler contains no surveys.", file=output)
        else:
            print("Surveys:", file=output)
        for tier_index, tier_surveys in enumerate(self.survey_lists):
            print(f"    Tier {tier_index}:", file=output)
            for survey in tier_surveys:
                print(f"        {survey}", file=output)
        print(f"camera: {camera}", file=output)
        print(f"nside: {self.nside}", file=output)
        print(f"rotator limits: {self.rotator_limits}", file=output)
        print(f"survey index: {self.survey_index}", file=output)
        result = output.getvalue()
        return result

    scheduler.__class__.__str__ = scheduler_str

    # Patching for BaseSurvey

    def survey_repr(self):
        return f"<{self.__class__.__name__} with survey_name='{self.survey_name}'>"

    rubin_sim.scheduler.surveys.BaseSurvey.__repr__ = survey_repr

    return scheduler
