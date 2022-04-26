"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the particles in a box generator class.
"""
from abc import ABC

import jax

from znrnd.core.data.data_generator import DataGenerator


class ConfinedParticles(DataGenerator, ABC):
    """
    A class to generate data for particles in a box.
    """

    def __init__(self, box_length: float = 2.0, dimension: int = 2.0):
        """
        Constructor for the ConfinedParticles data generator.

        Parameters
        ----------
        box_length : float
                Side length of box
        dimension : int
                Number of dimensions to consider.
        """
        self.box_length = box_length
        self.dimension = dimension

        self.data_pool = None

        self.key = jax.random.PRNGKey(0)

    def build_pool(self, n_points: int = 100):
        """
        Build a pool of data. Append to an existing one if it already exists.

        Parameters
        ----------
        n_points : int
                Number of points to add to the pool.

        Returns
        -------

        """
        self.data_pool = jax.random.uniform(
            self.key, (n_points, int(self.dimension)), minval=0, maxval=self.box_length
        )