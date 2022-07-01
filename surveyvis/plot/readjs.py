import importlib.resources


def read_javascript(fname):
    """Read javascript source code from the current package.

    Parameters
    ----------
    fname : `str`
        The name of the file from which to load js source code.

    Return
    ------
    js_code : `str`
        The loaded source code.
    """
    root_package = __package__.split(".")[0]

    try:
        # Standard python site-packages location
        with importlib.resources.path(root_package, ".") as root_path:
            js_path = root_path.joinpath("js", fname)
            with open(js_path, "r") as js_io:
                js_code = js_io.read()
    except FileNotFoundError:
        # infer we are in a project directory
        root_package = __package__.split(".")[0]

        with importlib.resources.path(root_package, ".") as root_path:
            # Check that our inference that we are in a
            # project directory is right
            assert root_path.parent.parent.joinpath("python", root_package).is_dir()

            full_name = root_path.parent.parent.joinpath("js", fname)
            with open(full_name, "r") as js_io:
                js_code = js_io.read()

    return js_code
