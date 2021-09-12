import cv2

#read in cam, maybe change some of the params
#cam = cv2.VideoCapture(0)  #for webcam
#cam=VideoStream(usePiCamera=1).start()

#cam.set(3,640) #width
#cam.set(4,480) #height
#cam.set(10,150) #brightness
class BirdID:
    def __init__(self):
        #initial settings
        confidenceThreshold = 0.3
        nms=True
        nms_threshold = 0.8 #lower value is more suppression
        # Read coco.names into the list, strip, clean etc
        classNames = []
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        # Import config file and weights
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        # create model, with paramters
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        return(net)

    def detect(self,img,net):
        #While loop to send image/cam to model, get ClassID (from classNames), confidences, bounding box params (A,B,C,D)
        #while True:
        #success,img = cam.read()
        classIDs, confs, boundingboxes = net.detect(img,confThreshold=confidenceThreshold)
        #print(classIDs,(confs))

        # ---------NonMaximumSuppression-----------
        if nms==True:
        #convert bounding boxes,confidences to np.lists for NMS
            bbox=list(boundingboxes)
            nmsconfs = list(confs.reshape(1,-1)[0])
            nmsconfs = list(map(float,nmsconfs))

            indices = cv2.dnn.NMSBoxes(bbox,nmsconfs,confidenceThreshold,nms_threshold)
            #print(indices)
            for i in indices:
                i=i[0]
                box=bbox[i]
                x,y,w,h = box[0],box[1],box[2],box[3]
                colourConfidence = nmsconfs[i]
                if classNames[classIDs[i][0] - 1] == 'bird':  # we only want to see birds
                    cv2.rectangle(img, (x, y), (x + w, h + y), color=(0,255*colourConfidence,255*(1-colourConfidence)), thickness=2)
                    cv2.putText(img, classNames[classIDs[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255*colourConfidence,255*(1-colourConfidence)), 2)
                    cv2.putText(img, str(round(nmsconfs[i] * 100, 1)) + "%", (box[0] + 10, box[1] + 50),cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255 * colourConfidence, 255 * (1 - colourConfidence)), thickness=2)

        else:
            if len(classIDs)!= 0:  #did we detect anything?
                #loop to get out results, draw box around it, label it
                for classID, confidence, box in zip(classIDs.flatten(),confs.flatten(),boundingboxes):
                    if classNames[classID-1]=='bird':#we only want to see birds
                        colourConfidence = confidence #(confidence - confidenceThreshold / confidence)  # use this to create a scale for colouring with confidence intervals?
                        cv2.rectangle(img,box,color=(0,255*colourConfidence,255*(1-colourConfidence)),thickness=2)
                        cv2.putText(img,classNames[classID-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255*colourConfidence,255*(1-colourConfidence)),thickness=2)
                        cv2.putText(img, str(round(confidence*100,1))+"%", (box[0] + 10, box[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255 * colourConfidence, 255 * (1 - colourConfidence)), thickness=2)
        return(img)

#cv2.imshow("output",img)
#cv2.waitKey(10)
