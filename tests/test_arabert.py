from cluling.arabic import AraBERT
#import numpy as np
import unittest
import os


class TestAraBERT(unittest.TestCase):

    dir_path        = os.path.join(os.path.dirname(os.path.realpath(__file__)), "toy-data")

    def test_example(self):
      """AraBERT should be of type 'AraBERT'
      """
      model = AraBERT()
      assert isinstance(model, AraBERT)

