import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import bokeh.plotting
import bokeh.io
import bokeh.document
from surveyvis.app.metric_maps import make_metric_figure, add_metric_app


class test_metric_maps(unittest.TestCase):
    def test_metric_maps(self):
        fig = make_metric_figure()
        with TemporaryDirectory() as dir:
            out_path = Path(dir)
            saved_html_fname = out_path.joinpath("test_page.html")
            bokeh.plotting.output_file(filename=saved_html_fname, title="Test Page")
            bokeh.plotting.save(fig)
            saved_png_fname = out_path.joinpath("test_fig.png")
            bokeh.io.export_png(fig, filename=saved_png_fname)

    def test_add_metric_app(self):
        doc = bokeh.document.document.Document()
        add_metric_app(doc)


if __name__ == "__main__":
    unittest.main()
