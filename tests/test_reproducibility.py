import random
import unittest

import numpy as np
import torch

from src.reproducibility import seed_everything


class SeedEverythingTests(unittest.TestCase):
    def test_reseeding_repeats_python_numpy_and_torch_random_values(self):
        seed_everything(17)
        first = (random.random(), np.random.random(), torch.rand(3))

        seed_everything(17)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_negative_seed_is_reproducible(self):
        seed_everything(-1)
        first = (random.random(), np.random.random(), torch.rand(3))

        seed_everything(-1)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
