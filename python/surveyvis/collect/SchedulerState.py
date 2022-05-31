import pickle
from collections import OrderedDict

import numpy as np

class SchedulerState:
    def __init__(self, input):
        self.scheds = {}
        self.conds = {}
        mjd_start_list = []
        
        def load_from_io(pio):
            while True:
                try:
                    sched, cond = pickle.load(pio)
                    mjd_start = cond.mjd_start
                    mjd_start_list.append(mjd_start)
                    self.scheds[mjd_start] = sched
                    self.conds[mjd_start] = cond 
                except EOFError:
                    break
        
        if hasattr(input, 'read'):
            load_from_io(input)    
        else:
            with open(input, "rb") as pio:
                load_from_io(pio)

        self.mjd_starts = np.sort(np.array(mjd_start_list))
        self._mjd = self.mjd_starts[0]
        self.survey_list_indexes = (1, 1)

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        mjd_start_index = np.searchsorted(self.mjd_starts, value, "left")
        if mjd_start_index > 0:
            self._mjd = self.mjd_starts[mjd_start_index - 1]
        else:
            raise ValueError

    @property
    def sched(self):
        return self.scheds[self.mjd]

    @property
    def cond(self):
        return self.conds[self.mjd]

    @property
    def basis_func(self):
        survey = self.sched.survey_lists[self.survey_list_indexes[0]][
            self.survey_list_indexes[1]
        ]
        basis_funcs = OrderedDict()
        for basis_func in survey.basis_functions:
            if hasattr(basis_func(self.cond), "__len__"):
                basis_funcs[basis_func.__class__.__name__] = basis_func
        return basis_funcs

    @property
    def healpix_map(self):
        maps = OrderedDict()
        for band in self.cond.skybrightness.keys():
            maps[f"{band}_sky"] = self.cond.skybrightness[band]

        for basis_func_key in self.basis_func.keys():
            maps[basis_func_key] = self.basis_func[basis_func_key](self.cond)

        return maps

