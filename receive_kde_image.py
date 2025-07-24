import os
import time
import cv2

# KDE Connect default downloads folder
downloads_folder = os.path.expanduser("~/Downloads")

print("Monitoring ~/Downloads for new images...")

# Track existing files
existing_files = set(os.listdir(downloads_folder))

while True:
    current_files = set(os.listdir(downloads_folder))
    new_files = current_files - existing_files

    for file in new_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
            image_path = os.path.join(downloads_folder, file)
            print(f"New image received: {image_path}")

            # Load and display image
            image = cv2.imread(image_path)
            cv2.imshow("Received Image", image)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()

    existing_files = current_files
    time.sleep(5)  # Check every 5 seconds
