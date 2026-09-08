"""Run: python -m unittest discover -s plugins/_sdk/example -p test_*.py"""
import unittest
from app.plugins.testing import FakeContext, sample_tracks
from __init__ import mean_major_strain


class CalculationTest(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(mean_major_strain(FakeContext().tracks()))

    def test_strain(self):
        self.assertAlmostEqual(mean_major_strain(sample_tracks()), (0 + .2 + .3) / 3)
