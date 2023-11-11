from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import os
import random

app = Flask(__name__)

video_cap = None

@app.route('/')
def index():
    return render_template('index.html')

from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import os
import random

app = Flask(__name__)

video_cap = None

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global video_cap

    while True:
        success, frame = video_cap.read()
        if not success:
            break
        else:
            # Draw a random rectangle on the frame
            draw_random_rectangle(frame)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame with the random rectangle
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_random_rectangle(frame):
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Define random rectangle parameters
    rect_color = (0, 255, 0)  # BGR color (green in this case)
    rect_thickness = 2
    rect_x = random.randint(0, width - 100)  # Random x-coordinate for the top-left corner
    rect_y = random.randint(0, height - 100)  # Random y-coordinate for the top-left corner
    rect_width = 40  # Rectangle width
    rect_height = 40  # Rectangle height

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, rect_thickness)

@app.route('/video', methods=['POST'])
def video():
    global video_cap

    if request.method == 'POST':
        f = request.files['file']

        if f:
            # Convert AVI to MP4
            avi_path = f"uploads/{f.filename}"
            f.save(avi_path)
            mp4_path = avi_path.replace('.avi', '.mp4')
            os.system(f'ffmpeg -i {avi_path} -c:v libx264 -preset medium -c:a aac -strict experimental -b:a 192k -movflags faststart {mp4_path}')

            # Use the MP4 file
            video_cap = cv2.VideoCapture(mp4_path)
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return "Error: No file provided."

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
