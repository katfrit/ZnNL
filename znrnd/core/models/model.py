"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the models.
"""
from typing import Callable

import jax.numpy as np


class Model:
    """
    Parent class for ZnRND Models.

    Attributes
    ----------
    model : Callable
            A callable class or function that takes a feature
            vector and returns something from it. Typically this is a
            neural network layer stack.
    """

    model: Callable

    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 10,
        batch_size: int = 1,
    ):
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : int
                Number of epochs to train over.
        batch_size : int
                Size of the batch to use in training.
        """
        raise NotImplemented("Implemented in child class.")

    def train_model_recursively(
        self, train_ds: dict, test_ds: dict, epochs: int = 10, batch_size: int = 1
    ):
        """
        Train a model recursively until a threshold is reached or the models fails
        to converge.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : int
                Number of epochs to train over.
        batch_size : int
                Size of the batch to use in training.
        """
        raise NotImplemented("Implemented in child class.")

    def compute_ntk(
        self, x_i: np.ndarray, x_j: np.ndarray = None, normalize: bool = True
    ):
        """
        Compute the NTK matrix for the model.

        Parameters
        ----------
        x_i : np.ndarray
                Dataset for which to compute the NTK matrix.
        x_j : np.ndarray (optional)
                Dataset for which to compute the NTK matrix.
        normalize : bool (default = True)
                If true, divide each row by its max value.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        raise NotImplemented("Implemented in child class")

    def __call__(self, feature_vector: np.ndarray):
        """
        Call the network.

        Parameters
        ----------
        feature_vector : np.ndarray
                Feature vector on which to apply operation.

        Returns
        -------
        output of the model.
        """
        self.model(feature_vector)