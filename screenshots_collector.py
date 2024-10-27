from typing import Literal, NoReturn
import pandas as pd
import pyautogui
import keyboard
import time
import os

DIR = "datasets/raw"

RES_X = 1920
RES_Y = 1080

df = pd.read_csv(filepath_or_buffer="datasets/data.csv") if os.path.exists(
    path="datasets/data.csv") else pd.DataFrame(columns=["image", "direction"])


def take_screenshot(timestamp: int) -> None:
    """ Taking the screenshot of the screen.

    Args:
        timestamp (int): The timestamp of the screenshot.
    """
    screenshot = pyautogui.screenshot(region=(405, 212, 849, 481))

    screenshot.save(fp=f"{DIR}/{timestamp}.png")


def save_data_row(timestamp: int, direction: str) -> None:
    """ Saving the data in the dataframe.

    Args:
        timestamp (int): The timestamp of the screenshot.
        direction (str): The direction of the screenshot.
    """
    df.loc[len(df)] = [f"{DIR}/{timestamp}.png", direction]


def save_data(df: pd.DataFrame) -> None:
    """ Saving the data in the CSV file.

    Args:
        df (DataFrame): The dataframe to save.
    """
    df = df.drop_duplicates()
    df.to_csv(path_or_buf="datasets/data.csv", index=False)


def check_directory() -> None:
    """ Checking if the directory exists or not. If not, then create the directory."""
    if not os.path.exists(path=DIR):
        os.makedirs(name=DIR)


def check_hotkey() -> None | Literal['up'] | Literal['down'] | Literal['left'] | Literal['right']:
    """ Checking if the user has pressed the hotkey or not.

    Returns:
        str: The direction of the screenshot.
    """
    if keyboard.is_pressed(hotkey="up"):
        return "up"
    elif keyboard.is_pressed(hotkey="down"):
        return "down"
    elif keyboard.is_pressed(hotkey="left"):
        return "left"
    elif keyboard.is_pressed(hotkey="right"):
        return "right"
    else:
        return None


def run_program() -> NoReturn:
    """ Running the program. """
    check_directory()

    while True:

        timestamp = int(time.time())

        direction = check_hotkey()

        if direction:
            take_screenshot(timestamp=timestamp)
            save_data_row(timestamp=timestamp, direction=direction)
            save_data(df=df)
            time.sleep(0.1)


if __name__ == "__main__":
    run_program()
