from flask import Flask, render_template, Response
import cv2
#import picamera as picamera

def gen_frames():
    while True:
        success, frame = camera.read()  #read in a camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

#init flask
app =Flask(__name__)
#init camera, for rpi use picamera command
camera = cv2.VideoCapture(0)
#camera = picamera.PiCamera(resolution='640x480', framerate=24)

#define template route
@app.route('/')
def index():
    return render_template('index.html')

#define route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)