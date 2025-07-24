import os
import time

downloads_folder = os.path.expanduser("~/Downloads")

print("Monitoring ~/Downloads for new images...")

# Track already existing files
existing_files = set(os.listdir(downloads_folder))

while True:
    current_files = set(os.listdir(downloads_folder))
    new_files = current_files - existing_files

    for file in new_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
            print(f"New image received: {file}")
            # You can add code here to process the image

    existing_files = current_files
    time.sleep(5)  # Check every 5 seconds
