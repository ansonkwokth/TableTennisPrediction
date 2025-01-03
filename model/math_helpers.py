import numpy as np
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

  
