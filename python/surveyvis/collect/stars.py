from collections import OrderedDict
import os
import pandas as pd

BSC5_URL = 'http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz'

def load_bright_stars(fname=None):
    if fname is None:
        fname = os.environ['BSC5_FNAME']
    
    ybs_columns = OrderedDict(
        (
            ("HR", (0, 4)),
            ("name", (4, 14)),
            ("RA_hour", (75, 77)),
            ("RA_min", (77, 79)),
            ("RA_sec", (79, 83)),
            ("decl_sign", (83, 84)),
            ("decl_deg", (84, 86)),
            ("decl_min", (86, 88)),
            ("decl_sec", (88, 90)),
            ("Vmag", (102, 107)),
        )
    )
    bs = pd.read_fwf(
        fname,
        colspecs=[ybs_columns[k] for k in ybs_columns],
        names=[k for k in ybs_columns],
    )
    bs["ra"] = (360 / 24) * (bs.RA_hour + (bs.RA_min + bs.RA_sec / 60.0) / 60.0)
    bs["decl"] = bs.decl_deg + (bs.decl_min + bs.decl_sec / 60.0) / 60.0
    southern_stars = bs.decl_sign == "-"
    bs.loc[southern_stars, "decl"] = -1 * bs.loc[southern_stars, "decl"]
    return bs
