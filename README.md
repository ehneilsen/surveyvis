# surveyvis
Tools for visualizing Rubin Observatory scheduler behaviour and LSST survey status

# Examination of scheduler snapshots

The application for examining scheduler snapshots depends on having the same
version of rubin_sim used for the snapshot. 

    $ conda create --name svistest1
    # canda activate svistest1
    $ conda install -c lsstts rubin-sim=0.8.0a2
    $ conda install bokeh pytest-flake8 pytest-black 
    $ pip install -e .

