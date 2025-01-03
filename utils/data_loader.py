import numpy as np
import pandas as pd
from math import comb
from typing import *
from datetime import datetime, timedelta


SPECIAL_NAME = ['Cho Seungmin', 'Wong Chun Ting', 'Lam Siu Hang', 'Xu Yingbin', 'Wang Lichen', 'Cho Ki Jeong']
INFO_COL = ['ID', 'Round', 'Datetime', 'Player', 'Game']



def find_round_title_idx(text_split: List[str]) -> Tuple[List[int], List[str]]:
    unique_names = ['-', 'WO', '', 'Awrd'] + SPECIAL_NAME
    round_indices = []
    titles = []

    for i, text in enumerate(text_split):
        if isinstance(text, str):
            try:
                int(text)  # Attempt to convert text to an integer
            except ValueError:
                # Only append if text has no '.' and is not in unique_names
                if '.' not in text and text not in unique_names:
                    # print(f"Found title: {text}")
                    round_indices.append(i)
                    titles.append(text)

    print()
    return round_indices, titles



def put_to_dt(text_split: List[str], idx: List[int]) -> Dict[int, List[str]]:
    """
    Extracts segments of text between specified indices and stores them in a dictionary.

    Args:
        text_split (List[str]): List of text items to extract segments from.
        idx (List[int]): List of indices marking the start of each segment.

    Returns:
        Dict[int, List[str]]: Dictionary where each key is an index and each value is a list of text items between indices.
    """
    dt = {}

    for i, index in enumerate(idx):
        if i != len(idx) - 1:
            dt[index] = text_split[index + 1:idx[i + 1]]
        else:
            dt[index] = text_split[index + 1:]

    return dt


    
def find_time_idx(text_split: List[str]) -> List[int]:
  """
  Finds indices in a list where the items contain a colon (':'),
  indicating a potential time entry.

  Args:
      text_split (List[str]): List of text items to search.

  Returns:
      List[int]: List of indices where items contain a colon.
  """
  time_indices = []

  for i, text in enumerate(text_split):
      if text and ':' in text:
          time_indices.append(i)

  return time_indices



def create_all_lt(dt_title: Dict[int, List[str]], text_split: List[str], year: int) -> List[List[Any]]:
    """
    Generates a list of game data entries based on provided title sections, split text, and a specified year.

    Args:
        dt_title (Dict[int, List[str]]): Dictionary with indices as keys and lists of text entries as values.
        text_split (List[str]): Original list of text items.
        year (int): Year for timestamp formatting.

    Returns:
        List[List[Any]]: List of game data entries, each containing game ID, title, time, and scores.
    """
    game_id = 0
    all_games = []

    for k, text_section in dt_title.items():
        # Find indices of time markers and split the section into time intervals
        time_indices = find_time_idx(text_section)
        dt_time = put_to_dt(text_section, time_indices)

        for kt, vt in dt_time.items():
            vt_array = np.array(vt)

            # Skip if any special markers are present
            if "WO" in vt_array or "Awrd" in vt_array:
                continue

            # Reshape and transpose scores
            score = vt_array.reshape(-1, 2).T
            timestamp = f"{year}.{text_section[kt]}"

            # Prepare two rows for each game
            row1 = [game_id, text_split[k], timestamp] + list(score[0, :])
            row2 = [game_id, text_split[k], timestamp] + list(score[1, :])

            # Skip games with exactly two score columns
            if score.shape[1] == 2:
                continue

            # Append rows to the list
            all_games.append(row1)
            all_games.append(row2)
            game_id += 1

    return all_games



def create_df(all_lt: List[List[Any]]) -> pd.DataFrame:
    """
    Converts a list of game data into a DataFrame with appropriately labeled columns.

    Args:
        all_lt (List[List[Any]]): List of game data entries.

    Returns:
        pd.DataFrame: DataFrame with labeled columns, including 'Date' and 'Time' derived from 'Datetime'.
    """

    # Determine the maximum length of rows to create the correct number of columns
    max_length = max(len(row) for row in all_lt)

    # Generate column names for sets based on the difference between max_length and INFO_COL length
    set_column_names = [f'Set{i + 1}' for i in range(max_length - len(INFO_COL))]

    # Create DataFrame with dynamic columns
    df = pd.DataFrame(all_lt, columns=INFO_COL + set_column_names)

    # Reverse the DataFrame for desired ordering
    df = df.iloc[::-1].reset_index(drop=True)

    # Convert 'Datetime' column to separate 'Date' and 'Time' columns
    df['Date'] = pd.to_datetime(df['Datetime'], format="%Y.%d.%m. %H:%M", errors='coerce').dt.date
    df['Time'] = pd.to_datetime(df['Datetime'], format="%Y.%d.%m. %H:%M", errors='coerce').dt.time

    return df[INFO_COL + ['Date', 'Time'] + set_column_names]



def create_df_from_text(text: str, year: int) -> pd.DataFrame:
    """
    Processes a text input to create a DataFrame with game data for a given year.

    Args:
        text (str): Raw text data containing game information.
        year (int): Year to use in date formatting.

    Returns:
        pd.DataFrame: DataFrame with structured game data.
    """

    # Split text by lines and remove any empty entries
    text_split = [line for line in text.split('\n') if line]

    # Find indices and titles for each round
    round_idx, titles_lt = find_round_title_idx(text_split)
    dt_title = put_to_dt(text_split, round_idx)

    # Create list of all game data entries and convert to DataFrame
    all_lt = create_all_lt(dt_title, text_split, year)
    df = create_df(all_lt)
    df = df.replace({None: np.nan})

    return df



def read_file(file_path: str) -> str:
    """Reads the entire contents of a file and returns it as a string."""
    with open(file_path, 'r') as file:
        return file.read()



def load_game_data(game: str, years: list, data_dir: str = './data') -> dict:
    """Loads game data for the specified game and years into a dictionary."""
    text_data = {}
    for year in years:
        file_path = f'{data_dir}/{game}{year}.txt'
        text_data[year] = read_file(file_path)

    return text_data



def create_game_dfs(game: str, years: list, text_data: dict) -> pd.DataFrame:
    """Creates DataFrames for each year based on the game type and concatenates them into a single DataFrame."""
    dfs = []
    for year in years:
        df = create_df_from_text(text_data[year], year)
        dfs.append(df)
    df_tot = pd.concat(dfs, ignore_index=True)
    df_tot.sort_values(by=['Date', 'Time'], inplace=True)
    df_tot.reset_index(drop=True, inplace=True)
    return df_tot




