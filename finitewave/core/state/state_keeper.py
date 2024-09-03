import os
import numpy as np

class StateKeeper:
    """Handles saving and loading the state of a simulation model.

    This class provides functionality to save and load the state of a simulation model, including
    all relevant variables specified in the model's `state_vars` attribute. It handles file operations
    for saving to and loading from numpy `.npy` files.

    Attributes
    ----------
    record_save : str
        Directory path where the simulation state will be saved.

    record_load : str
        Directory path from where the simulation state will be loaded.

    Methods
    -------
    save(model)
        Saves the state of the provided model to the specified directory.
    
    load(model)
        Loads the state from the specified directory and sets the state variables in the provided model.
    
    _save_variable(var_path, var)
        Helper method to save a variable to a numpy `.npy` file.
    
    _load_variable(var_path)
        Helper method to load a variable from a numpy `.npy` file.
    """

    def __init__(self):
        """
        Initializes the StateKeeper with default paths for saving and loading state.
        """
        self.record_save = ""
        self.record_load = ""

    def save(self, model):
        """
        Saves the state of the given model to the specified `record_save` directory.

        This method creates the necessary directories if they do not exist and saves each variable
        listed in the model's `state_vars` attribute as a numpy `.npy` file.

        Parameters
        ----------
        model : object
            The model object whose state is to be saved. The model must have a `state_vars` attribute
            listing the state variables to be saved.
        """
        if not os.path.exists(self.record_save):
            os.makedirs(self.record_save)
        for var in model.state_vars:
            self._save_variable(os.path.join(self.record_save, var + ".npy"),
                                model.__dict__[var])

    def load(self, model):
        """
        Loads the state from the specified `record_load` directory and sets it in the given model.

        This method loads each variable listed in the model's `state_vars` attribute from numpy `.npy`
        files and sets these variables in the model.

        Parameters
        ----------
        model : object
            The model object to which the state is to be loaded. The model must have a `state_vars` attribute
            which will be updated with the loaded variables.
        """
        for var in model.state_vars:
            setattr(model, var, self._load_variable(os.path.join(
                        self.record_load, var + ".npy")))

    def _save_variable(self, var_path, var):
        """
        Saves a variable to a numpy `.npy` file.

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
        Loads a variable from a numpy `.npy` file.

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
