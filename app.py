import time

from flask import Flask, render_template, Response
import cv2
import numpy as np
import picamera
import picamera.array
import io

def gen_frames():
    while True:
        global net, classNames
        #-------FOR PC------------
        #success, img = camera.read()  #read in a camera frame
        #------FOR RPI------------
        success=True
        img=camera
        classIDs, confs, bboxes = net.detect(img, confThreshold=thres)
        if len(classIDs) > 0:  # did we detect anything?
            # loop to get out results, draw box around it, label it
            for classID, confidence, box in zip(classIDs.flatten(), confs.flatten(), bboxes):
                if classNames[classID-1] == 'bird':  # we only want to see birds
                    colourConfidence = confidence  # (confidence - confidenceThreshold / confidence)  # use this to create a scale for colouring with confidence intervals?
                    cv2.rectangle(img, box, color=(0, 255 * colourConfidence, 255 * (1 - colourConfidence)),thickness=2)
                    cv2.putText(img, classNames[classID-1].upper(), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255 * colourConfidence, 255 * (1 - colourConfidence)),thickness=2)
                    cv2.putText(img, str(round(confidence * 100, 0)) + "%", (box[0] + 10, box[1] + 50),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255 * colourConfidence, 255 * (1 - colourConfidence)), thickness=2)
                else:
                    continue

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg',img)
            img=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

#init flask
app =Flask(__name__)
#init camera, for rpi use picamera command
#camera = cv2.VideoCapture(0)
buffer=io.BytesIO
with picamera.PiCamera() as picam:
    picam.rotation=180
    #picam.start_recording(buffer, format='mjpeg')
    picam.capture(buffer,format='jpeg')
    time.sleep(2)
    #with picamera.array.PiRGBArray(picam) as stream_obj:
        #picam.capture(stream_obj,format='bgr')
        #buffer = stream_obj.array
        #picam.stop_recording()

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
thres=0.5

#define template route
@app.route('/')
def index():
    return render_template('index.html')

#define route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="192.168.8.108", port=8000,debug=True)
    #app.run(debug=True)