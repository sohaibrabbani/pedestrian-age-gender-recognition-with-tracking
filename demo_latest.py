from datetime import datetime

import cv2
import imutils
import numpy as np
import time

classes = None
with open('C:/Users/sohai/Downloads/Anaconda Projects/YOLODetection/Latest/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
IGNORE = ["person"]
# generate different colors for different classes
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet('C:/Users/sohai/Downloads/Anaconda Projects/YOLODetection/yolov3_final.weights',
                      'C:/Users/sohai/Downloads/Anaconda Projects/YOLODetection/Latest/yolov3.cfg')

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('C:/Users/sohai/Downloads/Anaconda Projects/YOLODetection/video_4.mp4')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
frame_number = 0
imageCounter = 0
data = []
frame_start = 0
frame_end = 0
video_started = False
video_path = "C:/Users/sohai/Downloads/Anaconda Projects/YOLODetection/"
video_name = "video"
video_ext = ".avi"
frame_skip = 3
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	frame_number += 1
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.1:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				# idx = int(detection[0, 0, i, 1])

				if classes[classID] not in IGNORE:		#Add: Only select classes we want
					continue
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				if y >= (height/2):				#Add: For detection of a boundary in the frame
					if not video_started:
						frame_start = frame_number
						frame_end = frame_number + 80
						video_started = True
					# update our list of bounding box coordinates,
					# confidences, and class IDs
					#video_started = frame_end != frame_number
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(classes[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			if frame_skip > 2:
				crop_img = frame[y:y + h, x:x + w]
				imageCounter += 1
				current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				image_name = "object-detection-%d.jpg" % imageCounter
				cv2.imwrite(image_name, crop_img)
				data.append([current_time, video_path + image_name, video_path + video_name + video_ext])
				frame_skip = 0
			else:
				frame_skip += 1


	if frame_end >= frame_number:
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(video_path + video_name + video_ext, fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

			# some information on processing single frame
			if total > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					elap * total))

		# write the output frame to disk
		writer.write(frame)
	if frame_end == frame_number:
		video_started = False
		frame_start = 0
		frame_end = 0
		writer.release()
		writer = None
		video_name = "video%d" % frame_number


# release the file pointers
print("[INFO] cleaning up...")
vs.release()