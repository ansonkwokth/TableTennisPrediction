import numpy as np
from tqdm import tqdm

class RatingSystem:
    def __init__(self, params=None, learning_rate=0.01, base_param=(0, 1), update_sigma=True, verbose=False):
        """
        Initialize the rating system.

        :param params: Parameters for the players.
        :param learning_rate: Learning rate.
        """
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


    def _get_param(self, player):
        """
        Retrieve the current rating of a player. If the player does not exist, raise an error.

        :param player: The player's name or identifier.
        :return: Current rating of the player.
        """
        return self._params.get(player, self.base_param)



    def _add_sigma(self, sigma1, sigma2):
        return np.sqrt(sigma1**2 + sigma2**2)



    def expected_prob(self, param1, param2):
        """
        Calculate the expected prob. for Player A against Player B.

        :param param1: Parameters of Player 1.
        :param param2: Parameters of Player 2.
        :return: Expected prob for Player 1.
        """

        # Calculate the combined sigma and the difference in mu values
        sigma_tot = self._add_sigma(param1[1], param2[1])
        mu_diff = param1[0] - param2[0]
        z = mu_diff / sigma_tot

        # Calculate the predicted probability using the logistic function
        return 1 / (1 + np.exp(-z))



    def predict_point(self, player1, player2):
        """
        Predict the probability of Player 1 winning against Player 2 in a set.

        :param player1: Name or identifier of Player 1.
        :param player2: Name or identifier of Player 2.
        :return: Probability of Player A winning.
        """
        if self.verbose:
            for player in [player1, player2]:
                if player not in self._params: print(f'Player "{player}" not found. initialize param: {self.base_param}')

        param1 = self._get_param(player1)
        param2 = self._get_param(player2)
        return self.expected_prob(param1, param2)



    def _predict_set_config_from_p(self, p):
        q = 1 - p
        dt = {}

        # Calculate probabilities for the first 10+ points won by player 1
        for i in range(10):
            prob = comb(10 + i, 10) * p**11 * q**i
            dt[10 + i + 1] = prob  # Store probability for player 1 winning 11, 12, ... points
        # Calculate probability for the special case of winning exactly 22 points
        dt[22] = (1 / (1 - 2 * q * p)) * comb(20, 10) * p**12 * q**10

        return dt

        

    def predict_set_config(self, player1, player2):
        p = self.predict_point(player1, player2)
        return self._predict_set_config_from_p(p)


    def predict_set(self, player1, player2):
        return sum(self.predict_set_config(player1, player2).values())



    def _predict_game_config_from_p(self, p, n_win_sets=3):
        q = 1 - p
        dt = {}

        # Calculate probabilities for winning n_win_sets, n_win_sets+1, ...
        for i in range(n_win_sets):
            prob = comb(n_win_sets - 1 + i, n_win_sets - 1) * p**n_win_sets * q**i
            dt[n_win_sets + i] = prob  # Store probability for winning sets

        return dt

    def predict_game_config(self, player1, player2, n_win_sets=3):
        p = self.predict_set(player1, player2)
        return self._predict_game_config_from_p(p, n_win_sets)        



    def predict_game(self, player1, player2):
        return sum(self.predict_game_config(player1, player2).values())




    def add_player(self, player, param=None):
        """
        Add a new player with an optional custom rating.

        :param player: Name or identifier of the player.
        :param rating: Custom initial rating (defaults to base_rating if not provided).
        """
        if player not in self._params:
            self._params[player] = param if param is not None else self.base_param



    def display_params(self, round_digits=2):
        """
        Display the current ratings of all players, rounded to the specified number of digits.

        :param round_digits: Number of decimal places to round the parameters. Default is 2.
        """
        sorted_params = sorted(self._params.items(), key=lambda x: x[1][0], reverse=True)

        for player, param in sorted_params:
            rounded_param = tuple(round(value, round_digits) for value in param)
            print(f"{player}: {rounded_param}")





    def update_params(self, player1, player2, result1):
        """
        Update ratings for two players after a match.

        :param player1: Name or identifier of Player 1.
        :param player2: Name or identifier of Player 2.
        :param result1: Result for Player 1 (points1 / points_tot).
        """
        param1 = self._get_param(player1)
        param2 = self._get_param(player2)  # Fixed to get params for player2 instead of player1

        expected1 = self.expected_prob(param1, param2)
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

        :param dataset: A list of dictionaries or a DataFrame-like object with columns:
                        - 'player_a': Name or ID of Player A
                        - 'player_b': Name or ID of Player B
                        - 'result_a': Result for Player A (1 for win, 0.5 for draw, 0 for loss)
        """
        for matchi in tqdm(dataset):
            matchi = matchi.T

            player1, player2 = matchi[0]


            for seti in matchi[1:]:
                # seti = [int(si) for si in seti]
                points1, points2 = seti
                if np.isnan(points1) or np.isnan(points2): continue
                points_sum = points1 + points2
                if points_sum == 0: continue

                result1 = points1 / points_sum

                # # Add players to the system if they are not already in
                self.add_player(player1)
                self.add_player(player2)

                # # Update ratings based on the match result
                self.update_params(player1, player2, result1)




    def evaluate(self, dataset):
        verbose_ = self.verbose
        self.verbose = False
        correct = 0
        n_matches = 0

        history = []
        predictions = []
        for matchi in tqdm(dataset):
            matchi = matchi.T
            player1, player2 = matchi[0]

            p = self.predict_game(player1, player2)
            if p == 0.5: continue
            whowillwin = 0 if p > 0.5 else 1

            win1 = sum(matchi[1:, 0]>matchi[1:, 1])
            win2 = sum(matchi[1:, 0]<matchi[1:, 1])
            whowon = 0 if win1 > win2 else 1

            history.append(win1/(win1 + win2))
            predictions.append(p)

            n_matches += 1
            if (whowon == whowillwin): correct += 1
        self.verbose = verbose_

        acc = correct / n_matches
        print(f'\n === Accuracy: {acc} === \n')
        return acc, np.array(history), np.array(predictions)
