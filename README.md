# BirdCam #
RPi-based computer vision to look at birds visiting a bird feeder.
- - - -
### Plans ###

Current stable version: v1.2\
Current test version: \
Current dev version: v2

- [x] v1 Identifies when a bird visits the feeder, stream to local IP address
	- [x] v1.1 Need to utilise NMS to cleanup identification
	- [x] v1.2 Quality of life for testing: identify if running with webcam (windows PC) or deployed on RPi
- [ ] v2 Sets flag (why not `bird_is_present`?) when a bird is present and take a picture, add timestamp
- [ ] v3 Uses `bird_is_present==True` to send image (twitter bot?)  or otherwise notify people who're interested (maybe a buzzer? A light goes on when bird is present?)

- - - -

### References ###
* OpenCV-python
* __Using code hacked together from:__
	* "Object Detection Mobile Net SSD", *Computer Vision Zone*, <https://www.computervision.zone/courses/object-detection-mobile-net-ssd/>
	* "Video Streaming in Web Browsers with OpenCV & Flask", N. Lakhotia, _Towards Data Science_ , (**2020**), <https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00>
* Using Single-Shot multibox Detection (SSD) network "Mobile Net SSD v3" as supplied by "Object Detection Mobile Net SSD" course, *Computer Vision Zone*
