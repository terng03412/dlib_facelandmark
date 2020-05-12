from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

DRAW_IDXS = OrderedDict([("left_brow", (17,18,19,20,21)),
                        ("right_brow", (22,23,24,25,26)),
						("right_eye", (36,37,38,39,40,41)),
						("left_eye", (42,43,44,45,46,47)),
						("up_lib", (48,49,50,51,52,53,54,64,63,62,61,60)),
						("bot_lib", (60,59,58,57,56,55,64,65,66,67))

						 ])



# USAGE
# python predict_eyes.py --shape-predictor eye_predictor.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2


# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it to have a
	# maximum width of 400 pixels, and convert it to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	overlay = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	detections = detector(gray, 0)
	for k,d in enumerate(detections):
		shape = predictor(gray, d)
		for (_, name) in enumerate(DRAW_IDXS.keys()):
			pts = np.zeros((len(DRAW_IDXS[name]), 2), np.int32) 
			for i,j in enumerate(DRAW_IDXS[name]): 
				pts[i] = [shape.part(j).x, shape.part(j).y]

			pts = pts.reshape((-1,1,2))
			cv2.polylines(overlay,[pts],True,(0,255,0),thickness = 2)



	# show the frame
	cv2.imshow("Frame", overlay)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()