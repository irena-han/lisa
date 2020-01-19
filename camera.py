import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

#create a video object - 0 is used for external cameras
video = cv2.VideoCapture(1)
# Set up camera constants

print(cv2.__version__)

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
'''
while(True):
    check, frame = video.read()

    cv2.imshow("live", frame)

    end = cv2.waitKey(1)

    if end == ord('q'):
        break
'''
font = cv2.FONT_HERSHEY_SIMPLEX
classes = {'0': 'person'}
list_of_classes = [0]  #look for 0th index (person)
with tf.Session() as sess:
    sess.run(model.pretrained())
    #open webcam
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        img = cv2.resize(frame, (416, 416))
        imge = np.array(img).reshape(-1, 416, 416, 3)
        start_time = time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
        print("--- %s seconds ---" % (time.time() - start_time))  # to time it
        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 700, 700)

        # Define inside box coordinates (top left and bottom right)
        TL_inside = (int(416 * 0.1), int(416 * 0.35))
        BR_inside = (int(416 * 0.45), int(416 - 5))

        # Define outside box coordinates (top left and bottom right)
        TL_outside = (int(416 * 0.46), int(416 * 0.25))
        BR_outside = (int(416 * 0.8), int(416 * .85))

        #Draw in and out rectangle
        cv2.rectangle(frame, TL_outside, BR_outside, (255, 20, 20), 3)
        cv2.putText(frame, "Outside box", (TL_outside[0] + 10, TL_outside[1] - 10), font, 1, (255, 20, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, TL_inside, BR_inside, (20, 20, 255), 3)
        cv2.putText(frame, "Inside box", (TL_inside[0] + 10, TL_inside[1] - 10), font, 1, (20, 255, 255), 3, cv2.LINE_AA)

        boxes1 = np.array(boxes)
        for j in list_of_classes:  # iterate over classes
            count = 0
            if str(j) in classes:
                lab = classes[str(j)]
            if len(boxes1) != 0:
                # iterate over detected people
                for i in range(len(boxes1[j])):
                    box = boxes1[j][i]
                    # setting confidence threshold as 40%
                    if boxes1[j][i][4] >= .40:
                        count += 1

                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                        cv2.circle(frame, (x, y), 5, (75, 13, 180), -1)
                        cv2.putText(img, lab, (box[0], box[1]), font, 1.0, (0, 0, 255),lineType=cv2.LINE_AA)
            print(lab, ": ", count)

        # Display the output
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


inside_counter = 0
outside_counter = 0

pause = 0
pause_counter = 0

def personDetector(frame):
    # Use globals for the control variables so they retain their value after function exits
    global detected_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter

    # Draw boxes defining "outside" and "inside" locations.
    outRect = cv2.rectangle(frame, TL_outside, BR_outside, (255, 20, 20), 3)
    cv2.putText(frame, "Outside box", (TL_outside[0] + 10, TL_outside[1] - 10), font, 1, (255, 20, 255), 3, cv2.LINE_AA)
    inRect = cv2.rectangle(frame, TL_inside, BR_inside, (20, 20, 255), 3)
    cv2.putText(frame, "Inside box", (TL_inside[0] + 10, TL_inside[1] - 10), font, 1, (20, 255, 255), 3, cv2.LINE_AA)
