import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time
from twilio.rest import Client

#twilio account setup
account_sid = 'AC30c731ce1d27ad0dd41227d40f87bb4d'
auth_token = '2c6cba6308c0783210ee6e843383a820'
client = Client(account_sid, auth_token)

#Get the openCV version
print(cv2.__version__)

#Gets the Darknet model which is trained on the YOLOv3 model
inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

#sets the font for the boxes
font = cv2.FONT_HERSHEY_SIMPLEX

#Class keys are part of Darknet, where 0 corresponds to a person.
classes = {'0': 'person'}
list_of_classes = [0]

#start the camera session
with tf.Session() as sess:
    sess.run(model.pretrained())

    #create a video object - 0 is used for webcams, 1 for external cameras
    cap = cv2.VideoCapture(0)

    #While the camera is on
    while (cap.isOpened()):

        #Reads the frame and detects/displays relevant information
        ret, frame = cap.read()
        img = cv2.resize(frame, (416, 416))
        imge = np.array(img).reshape(-1, 416, 416, 3)
        start_time = time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
        print("--- %s seconds ---" % (time.time() - start_time))  # to time it
        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 700, 700)

        #Define zone coordinates
        badZoneInside = (int(416 * 0.1), int(416 * 0.35))
        badZoneOutside = (int(416 * 0.46), int(416 * 0.25))
        goodZoneInside = (int(416 * 0.45), int(416 - 5))
        goodZoneOutisde = (int(416 * 0.8), int(416 * .85))



        #Draw the rectangle zones
        cv2.rectangle(frame, badZoneOutside, goodZoneOutisde, (255, 20, 20), 3)
        cv2.putText(frame, "Outside box", (badZoneOutside[0] + 10, badZoneOutside[1] - 10), font, 1, (255, 20, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, badZoneInside, goodZoneInside, (20, 20, 255), 3)
        cv2.putText(frame, "Inside box", (badZoneInside[0] + 10, badZoneInside[1] - 10), font, 1, (20, 255, 255), 3, cv2.LINE_AA)

        #np array
        boxes1 = np.array(boxes)

        #iterates over the list of classes to identify objects
        for j in list_of_classes:
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

                        x = int(((boxes1[0][0][1] + boxes1[0][0][3]) / 2) * 416)
                        y = int(((boxes1[0][0][0] + boxes1[0][0][2]) / 2) * 416)

                        #draws a rectangle around the object, gets the center of the object via circle()
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                        cv2.circle(frame, (x, y), 5, (255,0,0), -1)
                        cv2.putText(img, lab, (box[0], box[1]), font, 1.0, (0, 0, 255),lineType=cv2.LINE_AA)
            print(lab, ": ", count)

        #Display the output
        cv2.imshow("image", img)

        #Exit upon a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#0 = bad zone/left box
#1 = in bed
zone = 0

#text alerts using twilio
if(zone):
    pass
else:
    message = client.messages.create(
        from_='+12016167238',
        body='Alexander Wright is off the bed',
        to='+16475451105'
    )

