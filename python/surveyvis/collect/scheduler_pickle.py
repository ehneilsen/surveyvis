import pickle
import os

__all__ = ["read_scheduler", "read_conditions"]

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

    with open(file_name, "rb") as pio:
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

    with open(file_name, "rb") as pio:
        _, conditions = pickle.load(pio)

    return conditions
