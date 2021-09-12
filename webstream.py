from py_image_search.bird_id import BirdID
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

outputFrame = None
lock = threading.Lock()  #inits output frame, locks to make sure threading works

#init flask
app=Flask(__name__)

#init video stream, let it warm up
cam=VideoStream(usePiCamer=1).start()
time.sleep(2.0)

#render html stuff
@app.route("/")
def index():
    return render_template("page.html")

def detect_bird():
    #grab global variables
    global cam, outputFrame, lock

    #init bird detection software
    bird=BirdID()
    while True:
        #loop over frames from video feed
        frame=cam.read()
        frame = imutils.resize(frame, width=400)
        timestamp=datetime.datetime.now()
        cv2.putText(frame,timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        bird_seen = bird.detect(frame,bird)
        with lock:
            outputFrame=frame.copy()
def generate():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag,encodedImage) = cv2.imencode(".jpg",outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device",default="192.168.8.108")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)",default=8000)
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_bird)
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
cam.stop()