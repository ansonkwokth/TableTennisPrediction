import numpy as np
from model.rating_system import RatingSystem



class ModifiedElo(RatingSystem):
    def __init__(self, learning_rate=0.1, base_param=(0, 1), update_sigma=True, binary=False, verbose=False):
        """
        Initialize the rating system.

        :param learning_rate: Learning rate.
        :param params: Parameters for the players.
        :param update_sigma: Update also sigma during the gradient descent.
        """
        super().__init__(learning_rate=learning_rate, binary=binary, verbose=verbose)

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

        self.params = {}
        self.base_param = base_param        
        self.update_sigma = update_sigma


        
    def __repr__(self):
        return f"EloLike(learning_rate={self.learning_rate}, base_param={self.base_param}, update_sigma={self.update_sigma}, verbose={self.verbose})"
       





    def _get_player_param(self, player):
        """
        Retrieve the current rating of a player. If the player does not exist, use the default.

        :param player: The player's name.
        :return: Current rating of the player.
        """
        return (player in self.params), self.params.get(player, self.base_param)




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



    def _prediction_verbose(self, player1, player2):
        if self.verbose:
            for player in (player1, player2):
                if player not in self.params: 
                    print(f'Player "{player}" not found. Initialize param: {self.base_param}')



    def _add_player(self, player, param=None):
        """
        Add a new player with an optional custom param.

        :param player: Name of the player.
        :param rating: Custom initial rating (defaults to base_rating if not provided).
        """
        if player not in self.params:
            self.params[player] = param if param is not None else self.base_param



    def display_params(self, round_digits=2, first_n=None):
        """
        Display the current ratings of all players, rounded to the specified number of digits.

        :param round_digits: Number of decimal places to round the parameters. Default is 2.
        :param first_n: If specified, only display the top `first_n` players based on their ratings.
        """
        # Sort players by the first element of their parameter tuple in descending order
        sorted_params = sorted(self.params.items(), key=lambda x: x[1][0], reverse=True)
        
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
        found_p1, param1 = self.get_player_param(player1)
        found_p2, param2 = self.get_player_param(player2)
        
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
        self.params[player1] = (
            param1[0] - self.learning_rate * mu1_update,
            param1[1] - self.learning_rate * sigma1_update
        )
        self.params[player2] = (
            param2[0] - self.learning_rate * mu2_update,
            param2[1] - self.learning_rate * sigma2_update
        )





