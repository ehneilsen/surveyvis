import unittest
from rubin_sim.scheduler.schedulers.core_scheduler import Core_scheduler
from rubin_sim.scheduler.features.conditions import Conditions
from surveyvis.collect.scheduler_pickle import read_scheduler, read_conditions

class test_scheduler_pickle(unittest.TestCase):
    
    def test_read_scheduler(self):
        scheduler = read_scheduler()
        self.assertIsInstance(scheduler, Core_scheduler)
        
    def test_read_conditions(self):
        conditions = read_conditions()
        self.assertIsInstance(conditions, Conditions)
        

if __name__ == "__main__":
    unittest.main()
