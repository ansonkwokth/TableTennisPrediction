import numpy as np
from tqdm import tqdm
from math import comb



class RatingSystem:

    def __init__(self, learning_rate=32, binary=False, verbose=False):
        
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(f"Expected 'learning_rate' to be an float/int, but got {type(learning_rate).__name__}")
        if learning_rate <= 0:
            raise ValueError(f"Expected 'learning_rate' to be positive, but got {learning_rate}")

        if not isinstance(verbose, bool):
            raise TypeError(f"Expected 'verbose' to be an bool, but got {type(verbose).__name__}")

        self.params = {} 
        self.learning_rate = learning_rate
        self.binary = binary
        self.verbose = verbose

        self._found_p1 = None
        self._found_p2 = None



    @staticmethod
    def _add_sigma(sigma1, sigma2):
        return np.sqrt(sigma1**2 + sigma2**2)



    @staticmethod
    def predict_set_config_from_p(p):
        q = 1 - p
        dt = {}
        # Calculate probabilities for the first 10+ points won by player 1
        for i in range(10):
            prob = comb(10 + i, 10) * p**11 * q**i
            dt[10 + i + 1] = prob  # Store probability for player 1 winning 11, 12, ... points
        # Calculate probability for the special case of winning exactly 22 points
        dt[22] = (1 / (1 - 2 * q * p)) * comb(20, 10) * p**12 * q**10
        return dt



    @staticmethod
    def predict_game_config_from_p(p, n_win_sets=3):
        # Ensure n_win_sets is odd
        if n_win_sets % 2 == 0:
            raise ValueError("'n_win_sets' must be an odd number.")
        q = 1 - p
        dt = {}
        # Calculate probabilities for winning n_win_sets, n_win_sets+1, ...
        for i in range(n_win_sets):
            prob = comb(n_win_sets - 1 + i, n_win_sets - 1) * p**n_win_sets * q**i
            dt[n_win_sets + i] = prob  # Store probability for winning sets
        return dt

  

    def get_player_param(self, player):
        return self._get_player_param(player)



    def _predict_point(self, player1, player2):
        """
        Predict the probability of Player 1 winning a point against Player 2.

        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning a point.
        """
        self._prediction_verbose(player1, player1)
        self._found_p1, param1 = self._get_player_param(player1)
        self._found_p2, param2 = self._get_player_param(player2)
        return self._expected_prob(param1, param2)

        
    def predict_point(self, player1, player2):
        p = self._predict_point(player1, player2)
        return self._found_p1, self._found_p2, p


    def predict_set_config(self, player1, player2):
        """
        Predict the probability of Player 1 winning a set against Player 2, 
        with total points being 'n', in the set. 
        For each possible 'n', calculate the corresponding prob.

        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: dict of probabilities of Player 1 winning the set with total 'n' points.
        """
        ps = self.predict_set_config_from_p(self._predict_point(player1, player2))
        return self._found_p1, self._found_p2, ps


    def predict_set(self, player1, player2):
        """
        Predict the probability of Player 1 winning a set against Player 2, 
        
        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning the set
        """
        _, _, ps = self.predict_set_config(player1, player2)
        return self._found_p1, self._found_p2, sum(ps.values())

    
    def predict_game_config(self, player1, player2, n_win_sets=3):
        """
        Predict the probability of Player 1 winning the game against Player 2, 
        with total number of sets being 'n', in the game. 
        For each possible 'n', calculate the corresponding prob.

        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :param n_win_sets: Number of winning sets to win the game
        :return: dict of probabilities of Player 1 winning the game with total 'n' sets.
        """
        _, _, p_set = self.predict_set(player1, player2)
        p = self.predict_game_config_from_p(p_set, n_win_sets) 
        return self._found_p1, self._found_p2, p


    def predict_game(self, player1, player2):
        """
        Predict the probability of Player 1 winning the game against Player 2, 
        
        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning the game
        """
        _, _, ps = self.predict_game_config(player1, player2)
        return self._found_p1, self._found_p2, sum(ps.values())



    def fit(self, dataset):
        """
        Fit the model to a dataset of matches.

        :param dataset: an array with shape (m, 2, s)
                        with m = number of matches
                             2 = 1v1
                             s = max number of sets in the dataset
                        Note that it must be sorted with time
                        for example:   
                        array([[['Reitspies D.', 11.0, 11.0, ..., nan, nan, nan],
                                ['Gavlas A.', 9.0, 9.0, ..., nan, nan, nan]],

                            [['Kleprlik J.', 3.0, 4.0, ..., nan, nan, nan],
                                ['Prokopcov D.', 11.0, 11.0, ..., nan, nan, nan]],

                            [['Horejsi M.', 11.0, 11.0, ..., 7.0, 9.0, nan],
                                ['Tregler T.', 4.0, 8.0, ..., 11.0, 11.0, nan]],
        """
        # loop over matches
        for matchi in tqdm(dataset, desc="Training model"):
            matchi = matchi.T
            # the first row are the players 
            player1, player2 = matchi[1]
            
            # loop over sets in the match
            for seti in matchi[2:]:
                points1, points2 = seti
                # skip nan entries
                if np.isnan(points1) or np.isnan(points2): continue
                # skip total points = 0
                points_sum = points1 + points2
                if points_sum == 0: continue

                result1 = points1 / points_sum
                if self.binary: result1 = result1 = 1 if result1 > 0.5 else 0

                # Add players to the system if they are not already in
                self._add_player(player1)
                self._add_player(player2)

                # Update ratings based on the match result
                self._update_params(player1, player2, result1)



