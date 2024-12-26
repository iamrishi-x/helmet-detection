import cv2 as cv
import numpy as np

# Initialize parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes and the YOLO model
classesFile = "/Users/user/Documents/Project/1_Face Detection/haarcascade/obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network
modelConfiguration = "/Users/user/Documents/Project/1_Face Detection/haarcascade/yolov3-obj.cfg"
modelWeights = "/Users/user/Documents/Project/1_Face Detection/haarcascade/yolov3-obj_2400.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    layerNames = net.getLayerNames()
    outLayers = net.getUnconnectedOutLayers()
    return [layerNames[i - 1] for i in outLayers] 

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    global classes
    label = '%.2f' % conf
    if classes:
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]),
                 (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    return 1

# Process the detected objects
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out = 0
    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # Process the indices if non-empty
    if indices and len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (tuple, list)) else idx  # Handle tuple or scalar
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # Display results if a target class is detected
    if frame_count_out > 0:
        cv.imshow('Detected', frame)
        cv.waitKey(1)

# Start capturing video
cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("No frame captured from video source.")
        break

    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Run the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

cap.release()
cv.destroyAllWindows()
