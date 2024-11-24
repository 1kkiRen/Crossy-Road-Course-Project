### %pip install inference-sdk dotenv 

from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
import cv2

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
"""Classes that are detected:
    hen
    car_left        // car moving left
    car_right       // car moving right
    obstacle        // bushes, stones, signs before train, etc.
    stop            // restart sign
    timber          // timbers on a river and green leves on river
    truck_right     // truck moving right
    truck_left      // truck moving left
"""

def detect_objects(image_need_to_be_detected):
    """
    Detects objects in an image using a specified model from Roboflow's inference API.
    This function takes an image as input, sends it to the Roboflow inference service, and retrieves 
    detection results including bounding boxes, class labels, confidence scores, and other metadata.
    Args:
        image (str): The path to the image file to be processed.
    Returns:
        dict: A dictionary containing the inference results, which includes:
            - inference_id (str): Unique identifier for the inference.
            - time (float): Time taken to perform the inference in seconds.
            - image (dict): Metadata about the image including width and height.
            - predictions (list of dict): A list of detected objects, where each object contains:
                - x (float): X-coordinate of the center of the bounding box.
                - y (float): Y-coordinate of the center of the bounding box.
                - width (float): Width of the bounding box.
                - height (float): Height of the bounding box.
                - confidence (float): Confidence score of the detection.
                - class (str): The class name of the detected object.
                - class_id (int): Numeric ID of the object class.
                - detection_id (str): Unique ID of the detection.
    Note:
        - Ensure the `ROBOFLOW_API_KEY` variable is properly defined and contains a valid API key for Roboflow.
        - The `model_id` parameter specifies which trained model to use for inference.
        - This function uses the Roboflow inference SDK, so make sure it is installed and properly configured.
    """
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    detections = CLIENT.infer(
        image_need_to_be_detected, model_id="crossy-road-rl-agent/2")
    return detections

#################### Example usage: ####################

## 1. To detect objects and get str result
if __name__ == "__main__":
    screenshot_path = "screens_for_detecting/test_screen.jpg"
    image = cv2.imread(screenshot_path)
    result = detect_objects(image)

## 2. To detect objects and get result with drawing
# import cv2
# def draw_bounding_boxes(image_path, predictions):
#     image = cv2.imread(image_path)
#     for prediction in predictions:
#         x_center, y_center, width, height = prediction['x'], prediction[
#             'y'], prediction['width'], prediction['height']
#         confidence = prediction['confidence']
#         class_name = prediction['class']
#         x_min = int(x_center - width / 2)
#         y_min = int(y_center - height / 2)
#         x_max = int(x_center + width / 2)
#         y_max = int(y_center + height / 2)
#         color = (0, 255, 0)
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
#         label = f"{class_name}: {confidence:.2f}"
#         cv2.putText(image, label, (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     # Display the image
#     cv2.imshow("Detected Objects", image)
#     key = cv2.waitKey(1000)  # Wait for 5 seconds or for a key press
#     if key == -1:  # If no key is pressed within the time
#         print("Closing window after timeout.")
#     cv2.destroyAllWindows()
#     # Optionally save the image
#     cv2.imwrite("annotated_image.jpg", image)

# screenshot_path = ""
# image = cv2.imread(screenshot_path)
# result = detect_objects(image)
# draw_bounding_boxes(screenshot_path, result['predictions'])
