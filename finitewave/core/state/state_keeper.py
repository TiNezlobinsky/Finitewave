import os
import numpy as np


class StateKeeper:
    """Handles saving and loading the state of a simulation model.

    This class provides functionality to save and load the state of a
    simulation model, including all relevant variables specified in the model's
    ``state_vars`` attribute. It handles file operations for saving to and
    loading from numpy files.

    Attributes
    ----------
    record_save : str
        Directory path where the simulation state will be saved.

    record_load : str
        Directory path from where the simulation state will be loaded.

    model : CardiacModel
        The model instance for which the state will be saved or loaded.
    """

    def __init__(self):
        self.record_save = ""
        self.record_load = ""
        self.model = None

    def initialize(self, model):
        """
        Initializes the state keeper with the given model.

        Parameters
        ----------
        model : CardiacModel
            The model instance for which the state will be saved or loaded.
        """
        self.model = model

    def save(self):
        """
        Saves the state of the given model to the specified ``record_save``
        directory.

        This method creates the necessary directories if they do not exist and
        saves each variable listed in the model's ``state_vars`` attribute as
        a numpy file.
        """
        if not os.path.exists(self.record_save):
            os.makedirs(self.record_save)

        for var in self.model.state_vars:
            self._save_variable(os.path.join(self.record_save, var + ".npy"),
                                self.model.__dict__[var])

    def load(self):
        """
        Loads the state from the specified ``record_load`` directory and sets
        it in the given model.

        This method loads each variable listed in the model's ``state_vars``
        attribute from numpy files and sets these variables in the model.
        """
        for var in self.model.state_vars:
            setattr(self.model, var, self._load_variable(os.path.join(
                    self.record_load, var + ".npy")))

    def _save_variable(self, var_path, var):
        """
        Saves a variable to a numpy file.

        Parameters
        ----------
        var_path : str
            The file path where the variable will be saved.

        var : numpy.ndarray
            The variable to be saved.
        """
        np.save(var_path, var)

    def _load_variable(self, var_path):
        """
        Loads a state variable from a numpy file.

        Parameters
        ----------
        var_path : str
            The file path from which the variable will be loaded.

        Returns
        -------
        numpy.ndarray
            The variable loaded from the file.
        """
        return np.load(var_path)
