import pickle
import os
import gzip
import importlib.resources

__all__ = ["read_scheduler", "read_conditions", "sample_pickle"]

try:
    PICKLE_FNAME = os.environ["SCHED_PICKLE"]
except KeyError:
    PICKLE_FNAME = None


def read_scheduler(file_name=None):
    """Read an instance of a scheduler object from a pickle.

    Parameters
    ----------
    file_name : `str`
        The name of the pickle file from which to load the scheduler.

    Returns
    -------
    scheduler : `rubin_sim.scheduler.schedulers.core_scheduler.Core_scheduler`
        An instance of a rubin_sim scheduler object.
    """
    if file_name is None:
        file_name = PICKLE_FNAME

    if file_name is None:
        file_name = sample_pickle()

    opener = gzip.open if file_name.endswith(".gz") else open

    with opener(file_name, "rb") as pio:
        scheduler, _ = pickle.load(pio)

    return scheduler


def read_conditions(file_name=None):
    """Read an instance of a conditions object from a pickle.

    Parameters
    ----------
    file_name : `str`
        The name of the pickle file from which to load the conditions.

    Returns
    -------
    conditions : `rubin_sim.scheduler.features.conditions.Conditions`
        An instance of a rubin_sim conditions object.
    """
    if file_name is None:
        file_name = PICKLE_FNAME

    if file_name is None:
        file_name = sample_pickle()

    opener = gzip.open if file_name.endswith(".gz") else open

    with opener(file_name, "rb") as pio:
        _, conditions = pickle.load(pio)

    return conditions


def sample_pickle():
    """Return the path of the sample pickle

    Returns
    -------
    fname : `str`
        File name of the sample pickle.
    """
    base_fname = 'scheduler.pickle.gz'
    root_package = __package__.split(".")[0]

    try:
        fname = str(importlib.resources.files(root_package).joinpath('data', base_fname))
    except AttributeError as e:
        # If we are using an older version of importlib, we need to do
        # this instead:
        if e.args[0] != "module 'importlib.resources' has no attribute 'files'":
            raise e

        with importlib.resources.path(root_package, ".") as root_path:
            fname = str(root_path.joinpath("data", base_fname))
                
    return fname
