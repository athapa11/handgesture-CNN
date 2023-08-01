import cv2
import numpy as np
import torch
from torchvision import models

# define the labels for output classes
classes = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 'rock', 'rock_on', 'scissor', 'thumbs', 'up']
# initialise class
model = models.vgg16(num_classes=len(classes))
# load pre-trained model weights
model.load_state_dict(torch.load('C:/Users/antho/PycharmProjects/CNN/vgg16.pth', map_location=torch.device('cpu'))
                      , strict=False)
model.eval()  # set dropout and batch normalisation layers to evaluation mode to prevent inconsistent inference results

# start the video feed
cam = cv2.VideoCapture(0)
num_frames = 0
while True:

    _, frame = cam.read()  # capture frame by frame
    frame = cv2.flip(frame, 1)  # to simulate mirrored image

    # coordinates for the region of interest ROI
    x1 = int(125 * frame.shape[1] / 240)
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(75 * frame.shape[1] / 195)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 255, 255), 1)  # draw the ROI
    roi = frame[y1:y2, x1:x2]  # extract the region of interest (ROI)

    cv2.resize(roi, (240, 195))
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)

    _, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresh", thresh) # show thresholded ROI

    # transform numpy array to a 4D tensor [batch, channel, height, width] for the model to use
    thresh = cv2.resize(thresh, (240, 195))
    img_arr = np.array(thresh)
    img_arr = np.stack((img_arr,) * 3, axis=0)
    img_arr = img_arr.astype(np.float32)
    print("img_arr: ", img_arr.shape)

    # transform numpy array to a 4D tensor [batch, channel, height, width] for the model to use
    img_t = torch.tensor(img_arr)
    img_t = img_t.unsqueeze(0)
    #print("tensor: ", img_t.shape)

    # Calling the predict method on model to predict the image
    output = model(img_t)
    _, pred = output.max(1)
    label = classes[pred[0]]
    print("label: {} \t prediction: {}".format(label, pred))  # results

    # set parameters for viewing the output class label on the cam
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (10, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2

    # add output class label to the video camera
    cv2.putText(frame,
                str(label),
                location,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    cv2.imshow("Capturing", frame)  # display frame
    key = cv2.waitKey(1)
    if key == ord('q'):  # press q to exit cam
        break
cam.release()
cv2.destroyAllWindows()
