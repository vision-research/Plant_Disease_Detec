from flask import Flask, request, send_file
import os

app = Flask(__name__)

# Folder to save received images
UPLOAD_FOLDER = "received_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"Image received and saved as {filepath}")

    return "Image received successfully!", 200

@app.route('/display', methods=['GET'])
def display_image():
    images = os.listdir(UPLOAD_FOLDER)
    if not images:
        return "No images received yet!", 404
    
    latest_image = os.path.join(UPLOAD_FOLDER, images[-1])
    return send_file(latest_image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
