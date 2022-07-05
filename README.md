# surveyvis
Tools for visualizing Rubin Observatory scheduler behaviour and LSST survey status

## Development state

The current code is in a early stage of development. The architecture will
be documented by [RTN-037](https://rtn-037.lsst.io/), currently little more than
an outline.

## Applications

There are presently two different applications in this project:

- `metric_maps`, a tool for visualizing metric output of MAF that takes the form
of a sky map.
- `sched_maps`, a tool for examing some elements of the state of objects used by
the scheduler, particularly those that take the form of a sky map, although some
indications of other elements are also present.

The project contains example data for each. At present, to use the example data,
different versions of dependencies are required, so the installation instructions
are slightly different in each case. (The reason for this is that the pickles
containing the sample objects to be viewed with `sched_maps` were created with
an old version of `rubin_sim`, and this older version needs to be installed for
these to be loaded.)

## Installation

First, get the code by cloning the github project:

    $ git clone git@github.com:ehneilsen/surveyvis.git

Go to the development directory, and download and decompress a data file used
by the automated tests. 

    $ cd surveyvis
    $ wget -O surveyvis/data/bsc5.dat.gz http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz
    $ gunzip surveyvis/data/bsc5.dat.gz

Create a `conda` environment with the appropriate dependencies, and activate it.
If you are running the `metric_maps` application, use the `conda` environment
file that includes a recent version of `survey_sim`:

    $ # ONLY IF RUNNING metric_maps
    $ conda env create -f environment.yaml
    $ conda activate surveyvis

If you are running `sched_maps`, get the one with the older version:

    $ # ONLY IF RUNNING sched_maps
    $ conda env create -f environment_080a2.yaml
    $ conda activate surveyvis080a2

Install the (development) `surveyvis` in your new environment:

    $ pip install -e .

Run the tests:

    $ pytest

## Running `metric_maps`

Activate the environment, and start the `bokeh` app. If `SURVEYVIS_DIR` is the
directory into which you cloned the `surveyvis` github repository, then:

    $ conda activate surveyvis
    $ bokeh serve ${SURVEYVIS_DIR}/surveyvis/app/metric_maps.py

The app will then give you the URL at which you can find the app. The data
displayed with the above instructions will be the sample metric map in the
project itself.

If you want to use a different data file, you can set the `METRIC_FNAME`
to its path before running the `bokeh` app. This is only a very short term
solution, and will be replaced soon.

## Running `sched_maps`

Activate the environment, and start the `bokeh` app. If `SURVEYVIS_DIR` is the
directory into which you cloned the `surveyvis` github repository, then:

    $ conda activate surveyvis080a2
    $ bokeh serve ${SURVEYVIS_DIR}/surveyvis/app/sched_maps.py

The app will then give you the URL at which you can find the app.