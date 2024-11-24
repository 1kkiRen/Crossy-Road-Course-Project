import cv2

button_template = cv2.imread("datasets/objects_screens/restart_button.png", cv2.IMREAD_GRAYSCALE)

if button_template is None:
    print("Error: 'restart_button.png' not found or could not be loaded.")
    exit(1)

def is_game_over(image, score_threshold=0.9):
    """
    Determines if the game is over by detecting the presence of a restart button in the provided image.

    This function uses template matching to locate a predefined restart button template in a specific 
    region of the input image. If the maximum match score exceeds the specified threshold, the game 
    is considered to be over.

    Args:
        image (numpy.ndarray): The input image, typically a screenshot of the game, in BGR format.
        score_threshold (float, optional): The threshold for the template matching score to consider 
            the restart button detected. Default is 0.9.
    Returns:
        bool: True if the restart button is detected in the image (indicating the game is over); 
            False otherwise.
    Steps:
        1. Converts the input image to grayscale.
        2. Crops a search box in the bottom-central region of the image to reduce the search area.
        3. Uses `cv2.matchTemplate` to find the similarity between the cropped region and the restart button template.
        4. Prints the maximum match score for debugging purposes.
        5. Compares the maximum score with the threshold and returns True if it exceeds the threshold, 
        indicating the restart button is detected.
    Note:
        - Ensure the `button_template` variable is properly initialized with the grayscale image of the 
        restart button template.
        - This function assumes that the restart button appears in the bottom-central region of the image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    cropped_search_box = image[int(h * 0.7):, int(w * 0.3):int(w * 0.7)]
    result = cv2.matchTemplate(cropped_search_box, button_template, cv2.TM_CCOEFF_NORMED)
    print(f"Max match score: {result.max()}")
    return result.max() > score_threshold

#################### Example usage: ####################
if __name__ == "__main__":
    screen_path = ""
    screen = cv2.imread(screen_path)
    print(f"Game over detected: {is_game_over(screen)}")