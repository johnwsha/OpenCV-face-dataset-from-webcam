# for webcam
# USAGE
# $ python3 build_face_dataset_webcam.py --output dataset/john

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())
path = args["output"]
path_arr = path.split('/')

# creating a folder for the images
if os.path.exists(path):
	print("Directory exists, continuing...")
	pass
else:
	try:
		# os.chdir(path_arr[0])
		os.mkdir(path)
		# os.chdir("..")
	except OSError:
	    print ("Creation of the directory %s failed" % path)
	else:
	    print ("Successfully created the directory %s " % path)

# load our serialized model from disk
print("[INFO] loading model...")

# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe(os.path.join(os.getcwd(), 'deploy.prototxt.txt'),
	os.path.join(os.getcwd(), 'res10_300x300_ssd_iter_140000.caffemodel'))

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml'))

# go into desired directory for saving images
# os.chdir(path_arr[0] + '/' + path_arr[1])
os.chdir(path)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(usePiCamera=True).start() # if using RPi3 camera!
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0 # for counting total number of images captured for building dataset
total_time = 0
count = 0

# loop over the frames from the video stream
while True:
	# timing each processing loop
	t0 = time.time()

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# printing processing time
	t1 = time.time()
	total_time += (t1 - t0)
	print("Processing time: " + str(t1 - t0))
	count += 1

	# if the `spacebar` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == 32:
		print("Image captured!")
		# p = os.path.sep.join([args["output"], "{}.png".format(
		# 	str(total).zfill(5))])
		cv2.imwrite("{}.png".format(str(total).zfill(5)), orig)
		total += 1

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
print("Average processing time: " + str(total_time / count))
print("Total processing time: " + str(total_time))
cv2.destroyAllWindows()
vs.stop()
