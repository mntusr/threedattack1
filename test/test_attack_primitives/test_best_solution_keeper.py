import unittest
from unittest import mock

from threedattack._attack_primitives import BestSolutionKeeper


class TestBestSolution(unittest.TestCase):
    def setUp(self):
        self.sampleBestFitness = (-1.1, "first")
        self.sampleSecondBestFitness = (-0.7, "second")
        self.sampleThirdBestFitness = (-0.1, "third")

    def test_update(self):
        sample = BestSolutionKeeper[str]()

        sample.update(*self.sampleSecondBestFitness)
        best1 = sample.get_best()
        self.assertEqual(best1, self.sampleSecondBestFitness[1])

        sample.update(*self.sampleBestFitness)
        best2 = sample.get_best()
        self.assertEqual(best2, self.sampleBestFitness[1])

        sample.update(*self.sampleThirdBestFitness)
        best3 = sample.get_best()
        self.assertEqual(best3, self.sampleBestFitness[1])

    def test_get_or_fail_fail(self):
        sample = BestSolutionKeeper[str]()
        with self.assertRaises(Exception):
            sample.get_best()
