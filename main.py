from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import cvzone
from ultralytics import YOLO
import pyttsx3

app = Flask(__name__)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load YOLO model
objectModel = YOLO("yolov10n.pt")

# COCO classes
classNames = objectModel.names

# Restricted items list (add as needed)
restricted_items = ["knife", "scissors", "baseball bat", "bottle", "wine glass", "gold", "cell phone"]

# To avoid repeating speech
already_announced = set()

@app.route('/')
def home():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        # Here, you can add code to save the data to a database
        print(f"Received sign-up data: Name={name}, Email={email}")
        return f"Thank you for signing up, {name}!"
    return render_template('signup.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
     while True:
        frameCaptured, frame = cap.read()
        if not frameCaptured:
            break

        # YOLO inference
        results = objectModel(frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy().astype("int")
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = classNames[class_id]

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)
                    cvzone.putTextRect(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), scale=1, thickness=1)

                    # Check if label is in restricted items list
                    if label in restricted_items and label not in already_announced:
                        # Speak warning
                        announcement = f"{label} is not allowed in the airport"
                        print(announcement)
                        engine.say(announcement)
                        engine.runAndWait()
                        already_announced.add(label)  # prevent repeating

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
