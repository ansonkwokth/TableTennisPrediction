import numpy as np
import copy
from collections import defaultdict
from model.rating_system import RatingSystem



class BaggingRatingSystem(RatingSystem):
    def __init__(self, estimator, n_models, sample_ratio=0.8, random_state=0, verbose=False):
        """
        Initialize the EnsembleModel.

        Args:
            base_model_class (class): The class of the base model to use.
            n_models (int): The number of models in the ensemble.
        """
        super().__init__(verbose=verbose)

        self.estimator = estimator
        self.n_models = n_models
        self.models = [copy.deepcopy(estimator) for _ in range(n_models)]
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)

        self._merged_estimator = copy.deepcopy(estimator)



 
    def __repr__(self):
        return f"BaggingRatingSystem(estimator={self.estimator}, n_models={self.n_models}, sample_ratio={self.sample_ratio}, random_state={self.random_state})"
       



    def _sample_random_data(self, data):
        """
        Randomly sample the input data from data by shuffling game IDs,
        ensuring all rows with the same game ID are kept together.

        Args:
            data (numpy array): Input data of shape (n_games, 2, n_sets).
            sample_ratio (float): Proportion of data to be sampled (default is 0.8).

        Returns:
            tuple: sample_data (numpy array): Sampled data
        """

        # Extract unique game IDs
        game_ids = np.unique(data[:, 0, 0].astype(int))

        # Shuffle game IDs
        np.random.shuffle(game_ids)

        # Split game IDs into train and test
        split_index = int(self.sample_ratio * len(game_ids))
        sample_ids = game_ids[:split_index]
        # Create sampled datasets based on the IDs
        sample_data = data[np.isin(data[:, 0, 0].astype(int), sample_ids)]
        return sample_data




    def fit(self, data):
        """
        Train each model on its respective data split.

        Args:
            data_splits (list): List of data splits for training.
        """
        models_params = []
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}...")
            data_splits = self._sample_random_data(data)
            model.fit(data_splits)
            models_params.append(model.params)
            
        merged_params = self._merge_and_average(models_params)
        self._create_merged_estimator(merged_params)



    def _merge_and_average(self, model_outputs):
        # Create a defaultdict to accumulate player predictions (values can be either numeric or tuples)
        merged_predictions = defaultdict(list)

        # Accumulate predictions for each player
        for model in model_outputs:
            for player, value in model.items():
                merged_predictions[player].append(value)

        # Calculate the average for each player, handling tuples and numeric values
        final_predictions = {}
        for player, values in merged_predictions.items():
            # If the values are tuples (e.g., (score, score)), we need to handle it differently
            if isinstance(values[0], tuple):
                # Average the elements inside the tuple (handle each element separately)
                element_averages = tuple(np.mean([v[i] for v in values]) for i in range(len(values[0])))
                final_predictions[player] = element_averages
            else:
                # Regular averaging for floats (simple score averaging)
                final_predictions[player] = np.mean(values) if len(values) > 1 else values[0]
        return final_predictions



    def _create_merged_estimator(self, params):
        self._merged_estimator.params = params
        self._get_player_param = self._merged_estimator._get_player_param
        self._expected_prob = self._merged_estimator._expected_prob
        self._prediction_verbose = self._merged_estimator._prediction_verbose



