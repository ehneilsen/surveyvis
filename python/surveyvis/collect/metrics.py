import importlib.resources
from os import environ

__all__ = ["get_metric_path"]

def get_metric_path():
    """Get the path to a file with numpy metrics
    
    Returns
    -------
    metric_path : `str`
        The path to the file containing the MAF metric
    """
    if 'PICKLE_FNAME' in environ:
        return environ['PICKLE_FNAME']
    
    root_package = __package__.split(".")[0]
    base_fname = 'baseline_v2_0_10yrs_CoaddM5_r_HEAL.npz'
    with importlib.resources.path(root_package + '.data', base_fname) as pickle_path:
        metric_values_fname = str(pickle_path)
    
    return metric_values_fname
    