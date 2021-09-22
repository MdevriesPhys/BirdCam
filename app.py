import time
from flask import Flask, render_template, Response
import cv2
import numpy as np
import io
import platform
import socket

def gen_frames():
    while True:
        global net, classNames, camera
        #-------FOR WEBCAM------------
        if test_mode==True:
            success, img = camera.read()  #read in a camera frame
        #------FOR RPI------------
        else:
            import picamera
            import picamera.array
            success=True
            with picamera.PiCamera() as picam:
                picam.rotation = 180   #rotation of camera image
                time.sleep(0.1)
                with picamera.array.PiRGBArray(picam) as stream_obj:
                    picam.capture(stream_obj, format='bgr')
                    camera = stream_obj.array
            img=camera
        #------------------------------------------------

        classIDs, confs, bboxes = net.detect(img, confThreshold=thres)
        #------------NMS utilisation----------------
        bboxes = list(bboxes)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))

        indices = cv2.dnn.NMSBoxes(bboxes,confs,thres,nms_threshold)

        if len(indices)>0: #did we detect anything?
            for i in indices:#loop to get out results, draw box around it, label it
                i = i[0]
                if classNames[classIDs[i][0] - 1] == target_item:  # we only want to see the looked for item
                    box = bboxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    confidence=confs[i]
                    # colourConfidence = confidence  # (confidence - confidenceThreshold / confidence)  # use this to create a scale for colouring with confidence intervals?
                    colourConfidence = 1  # bright green, no changing with confidence level
                    cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255 * colourConfidence, 255 * (1 - colourConfidence)),thickness=2)
                    #cv2.putText(img, classNames[classIDs[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, classNames[classIDs[i][0] - 1].upper(), (x + 10, y + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255 * colourConfidence, 255 * (1 - colourConfidence)),thickness=2)
                    cv2.putText(img, str(round(confidence * 100, 0)) + "%", (x + 10, y + 50),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255 * colourConfidence, 255 * (1 - colourConfidence)), thickness=2)
                else:
                    continue

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg',img)
            img=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

#--------startups------------
app =Flask(__name__)
test_mode = False #assume rpi deployment
target_item = "bird" #looking for birds
if platform.system()=="Windows":
    test_mode=True  #this is a flag to decide between testing on windows or deployed on RPi
    camera = cv2.VideoCapture(0)  #inits webcam camera, for testing
    target_item = "cell phone" #debug using phone, easier to use

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
thres=0.25
nms_threshold=0.2

#define template route
@app.route('/')
def index():
    return render_template('index.html')

#define route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if test_mode==True:
    app.run(debug=True)
else:
    hostname=socket.gethostname()
    local_ip=socket.gethostbyname(hostname)
    app.run(host=str(local_ip), port=8000,debug=True)