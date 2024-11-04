import os
import cv2
from MTM import matchTemplates

os.makedirs('output', exist_ok=True)

screenshot_dir = "screens_for_detecting"
template_dir = "objects_screens"

# Load all the template images
template_list = []
for filename in os.listdir(template_dir):
    template_path = os.path.join(template_dir, filename)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template_img is None:
        print(f"Error loading template: {template_path}")
        continue

    template_list.append((filename.split('.')[0], template_img))

# Iterate over each screenshot in the screenshot directory
for screenshot_file in os.listdir(screenshot_dir):
    screenshot_path = os.path.join(screenshot_dir, screenshot_file)
    screenshot = cv2.imread(screenshot_path)

    # Check if the screenshot was loaded correctly
    if screenshot is None:
        print(f"Error loading screenshot: {screenshot_path}")
        continue

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Perform Multi-Template Matching on the screenshot
    hits = matchTemplates(
        template_list,
        screenshot_gray,
        score_threshold=0.60,
        searchBox=(0, 0, screenshot.shape[1], screenshot.shape[0]),
        method=cv2.TM_CCOEFF_NORMED,
        maxOverlap=0.1
    )

    # Create a copy of the screenshot to draw on
    overlay = screenshot.copy()

    # Draw bounding boxes and labels
    for label, bbox, score in hits:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Convert overlay to RGB for displaying in matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


    # Save the overlay image with bounding boxes
    output_file_path = os.path.join('output', f'detected_{screenshot_file}')
    cv2.imwrite(output_file_path, overlay)
    # print(f"Overlay image saved to '{output_file_path}'")