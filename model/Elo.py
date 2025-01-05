import numpy as np
from model.rating_system import RatingSystem



class Elo(RatingSystem):
    def __init__(self, learning_rate=32, base_rate=1000, binary=False, verbose=False):
        """
        Initialize the Elo rating system.

        :param learning_rate: Learning rate.
        :param params: Rate for the players.
        """
        super().__init__(learning_rate=learning_rate, binary=binary, verbose=verbose)

        if not isinstance(base_rate, (float, int)):
            raise TypeError(f"Expected 'base_rate' to be an float/int, but got {type(base_rate).__name__}")
        
        if not isinstance(verbose, bool):
            raise TypeError(f"Expected 'verbose' to be an bool, but got {type(verbose).__name__}")

        self.params = {} 
        self.base_rate = base_rate



    def __repr__(self):
        return f"Elo(learning_rate={self.learning_rate}, base_rate={self.base_rate}, verbose={self.verbose})"






    def _get_player_param(self, player):
        """
        Retrieve the current rating of a player. If the player does not exist, use the default.

        :param player: The player's name.
        :return: Current rating of the player.
        """
        return (player in self.params), self.params.get(player, self.base_rate)



    def _expected_prob(self, param1, param2):
        """
        Calculate the expected prob. of winning a point.

        :param param1: Parameters of Player 1.
        :param param2: Parameters of Player 2.
        :return: Expected prob for Player 1.
        """
        # Calculate the combined sigma and the difference in mu values
        z = (param1 - param2) / 400
        # Calculate the predicted probability using the logistic function
        return 1 / (1 + 10**(-z))


    
    def _prediction_verbose(self, player1, player2):
        if self.verbose:
            for player in (player1, player2):
                if player not in self.params: 
                    print(f'Player "{player}" not found. Initialize param: {self.base_rate}')



    def _add_player(self, player, rate=None):
        """
        Add a new player with an optional custom param.

        :param player: Name of the player.
        :param rating: Custom initial rating (defaults to base_rating if not provided).
        """
        if player not in self.params:
            self.params[player] = rate if rate is not None else self.base_rate



    def display_params(self, first_n=None):
        """
        Display the current ratings of all players, rounded to the specified number of digits.

        :param first_n: If specified, only display the top `first_n` players based on their ratings.
        """
        # Sort players by the first element of their parameter tuple in descending order
        sorted_params = sorted(self.params.items(), key=lambda x: x[1], reverse=True)
        
        # Limit the list to the first `first_n` players if specified
        if first_n is not None:
            sorted_params = sorted_params[:first_n]

        # Display the parameters
        for player, rate in sorted_params:
            print(f"{player}: {rate}")



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
        mu_diff = param1 - param2

        # Calculate updates
        mu1_update = (result1 - expected1)
        mu2_update = ((1-result1) - expected1)
        
        # Update ratings and assign new tuples
        self.params[player1] = param1 + self.learning_rate * mu1_update
        self.params[player2] = param2 + self.learning_rate * mu2_update
        



