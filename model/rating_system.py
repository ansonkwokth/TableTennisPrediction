import numpy as np
from tqdm import tqdm
from math import comb


class ProbCalculator:
    
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





class EloLike(ProbCalculator):
    def __init__(self, learning_rate=0.01, base_param=(0, 1), update_sigma=True, verbose=False):
        """
        Initialize the rating system.

        :param params: Parameters for the players.
        :param learning_rate: Learning rate.
        """
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(f"Expected 'learning_rate' to be an float/int, but got {type(learning_rate).__name__}")

        if learning_rate <= 0:
            raise ValueError(f"Expected 'learning_rate' to be positive, but got {learning_rate}")

        if not isinstance(base_param, tuple):
            raise TypeError(f"Expected 'base_param' to be a tuple, but got {type(base_param).__name__}")
        if len(base_param) != 2:
            raise ValueError(f"Expected 'value' to be a 2-entry tuple, but got {len(base_param)} entries")
        if not all(isinstance(entry, (int, float)) for entry in base_param):
            raise TypeError("Both entries in 'base_param' must be float/int")
        if base_param[1] <= 0:
            raise ValueError(f"The second entry in 'base_param' must be greater than 0, but got {base_param[1]}")
        
        if not isinstance(update_sigma, bool):
            raise TypeError(f"Expected 'update_sigma' to be an bool, but got {type(update_sigma).__name__}")

        if not isinstance(verbose, bool):
            raise TypeError(f"Expected 'verbose' to be an bool, but got {type(verbose).__name__}")

        self._params = {} 
        self.learning_rate = learning_rate
        self.base_param = base_param
        self.verbose = verbose
        self.update_sigma = update_sigma



    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, input_params):
        self._params = input_params



    def get_player_param(self, player):
        """
        Retrieve the current rating of a player. If the player does not exist, use the default.

        :param player: The player's name.
        :return: Current rating of the player.
        """
        return self._params.get(player, self.base_param)



    def _expected_prob(self, param1, param2):
        """
        Calculate the expected prob. of winning a point.

        :param param1: Parameters of Player 1.
        :param param2: Parameters of Player 2.
        :return: Expected prob for Player 1.
        """
        # Calculate the combined sigma and the difference in mu values
        z = (param1[0] - param2[0]) / self._add_sigma(param1[1], param2[1])
        # Calculate the predicted probability using the logistic function
        return 1 / (1 + np.exp(-z))



    def predict_point(self, player1, player2):
        """
        Predict the probability of Player 1 winning a point against Player 2.

        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning a point.
        """
        if self.verbose:
            for player in (player1, player2):
                if player not in self._params: 
                    print(f'Player "{player}" not found. Initialize param: {self.base_param}')

        param1 = self.get_player_param(player1)
        param2 = self.get_player_param(player2)
        return self._expected_prob(param1, param2)



    def predict_set_config(self, player1, player2):
        """
        Predict the probability of Player 1 winning a set against Player 2, 
        with total points being 'n', in the set. 
        For each possible 'n', calculate the corresponding prob.

        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: dict of probabilities of Player 1 winning the set with total 'n' points.
        """
        p = self.predict_point(player1, player2)
        return self.predict_set_config_from_p(p)

    def predict_set(self, player1, player2):
        """
        Predict the probability of Player 1 winning a set against Player 2, 
        
        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning the set
        """
        return sum(self.predict_set_config(player1, player2).values())



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
        p = self.predict_set(player1, player2)
        return self.predict_game_config_from_p(p, n_win_sets)        

    def predict_game(self, player1, player2):
        """
        Predict the probability of Player 1 winning the game against Player 2, 
        
        :param player1: Name of Player 1.
        :param player2: Name of Player 2.
        :return: Probability of Player 1 winning the game
        """
        return sum(self.predict_game_config(player1, player2).values())



    def _add_player(self, player, param=None):
        """
        Add a new player with an optional custom param.

        :param player: Name of the player.
        :param rating: Custom initial rating (defaults to base_rating if not provided).
        """
        if player not in self._params:
            self._params[player] = param if param is not None else self.base_param



    def display_params(self, round_digits=2, first_n=None):
        """
        Display the current ratings of all players, rounded to the specified number of digits.

        :param round_digits: Number of decimal places to round the parameters. Default is 2.
        :param first_n: If specified, only display the top `first_n` players based on their ratings.
        """
        # Sort players by the first element of their parameter tuple in descending order
        sorted_params = sorted(self._params.items(), key=lambda x: x[1][0], reverse=True)
        
        # Limit the list to the first `first_n` players if specified
        if first_n is not None:
            sorted_params = sorted_params[:first_n]

        # Display the parameters
        for player, param in sorted_params:
            rounded_param = tuple(round(value, round_digits) for value in param)
            print(f"{player}: {rounded_param}")




    def _update_params(self, player1, player2, result1):
        """
        Update ratings for two players after a match.

        :param player1: Name or identifier of Player 1.
        :param player2: Name or identifier of Player 2.
        :param result1: Result for Player 1 (points1 / points_tot).
        """
        param1 = self.get_player_param(player1)
        param2 = self.get_player_param(player2)  
        
        expected1 = self._expected_prob(param1, param2)
        mu_diff = param1[0] - param2[0]
        sigma_tot = self._add_sigma(param1[1], param2[1])

        # Calculate updates
        mu1_update = (expected1 - result1) / sigma_tot
        mu2_update = - (expected1 - result1) / sigma_tot
        sigma1_update = sigma2_update = 0
        if self.update_sigma:
            sigma1_update = (-mu_diff * param1[1]) / sigma_tot**3 * (expected1 - result1)
            sigma2_update = (-mu_diff * param2[1]) / sigma_tot**3 * (expected1 - result1)

        # Update ratings and assign new tuples
        self._params[player1] = (
            param1[0] - self.learning_rate * mu1_update,
            param1[1] - self.learning_rate * sigma1_update
        )
        self._params[player2] = (
            param2[0] - self.learning_rate * mu2_update,
            param2[1] - self.learning_rate * sigma2_update
        )



    def fit(self, dataset):
        """
        Fit the model to a dataset of matches.

        :param dataset: an array with shape (m, 2, s)
                        with m = number of matches
                             2 = 1v1
                             s = max number of sets in the dataset
                        for example:   
                        array([[['Reitspies D.', 11.0, 11.0, ..., nan, nan, nan],
                                ['Gavlas A.', 9.0, 9.0, ..., nan, nan, nan]],

                            [['Kleprlik J.', 3.0, 4.0, ..., nan, nan, nan],
                                ['Prokopcov D.', 11.0, 11.0, ..., nan, nan, nan]],

                            [['Horejsi M.', 11.0, 11.0, ..., 7.0, 9.0, nan],
                                ['Tregler T.', 4.0, 8.0, ..., 11.0, 11.0, nan]],
        """
        # loop over matches
        for matchi in tqdm(dataset):
            matchi = matchi.T
            # the first row are the players 
            player1, player2 = matchi[0]
            
            # loop over sets in the match
            for seti in matchi[1:]:
                points1, points2 = seti
                # skip nan entries
                if np.isnan(points1) or np.isnan(points2): continue
                # skip total points = 0
                points_sum = points1 + points2
                if points_sum == 0: continue

                result1 = points1 / points_sum

                # Add players to the system if they are not already in
                self._add_player(player1)
                self._add_player(player2)

                # Update ratings based on the match result
                self._update_params(player1, player2, result1)




