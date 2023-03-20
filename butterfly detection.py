#!/usr/bin/env python
# coding: utf-8

# In[13]:


#pip install opencv-contrib-python==4.5.5.62


# In[1]:


#conda install tensorflow 


# In[2]:


#pip install pandas


# In[5]:


#pip install pillow


# In[7]:


#pip install matplotlib


# In[1]:


import os
import numpy as np
import pandas as pd
import PIL
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
import cv2


# In[7]:


net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

cap = cv2.VideoCapture("Lots of Butterfly Flying in Flowers Garden  How Butterflies Pollinate Flowers.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))


# In[8]:


result = None

while(cap.isOpened()):
        
    for i in range(fps):
        ret, frame = cap.read()
        
    
    if not ret:
        break
        
    if result is None:
    # Constructing code of the codec to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # object to save video
        result = cv2.VideoWriter('./butterfly_detected.mp4', fourcc, 1, (frame.shape[1], frame.shape[0]), True)
    
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)

    #Extract the detected butterfly objects from the output
    class_ids = []
    confidences = []
    boxes = []
    colors = [(0,0,255),(0,255,255),(255,255,0), (255,255,255), (255,0,255),(0,255,0),(0,0,0),(255,0,0)]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 56:  # 56 is the class ID of butterfly in COCO dataset
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = center_x - w//2
                y = center_y - h//2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    #Initialize the tracker for each detected butterfly object
    trackers = cv2.legacy.MultiTracker_create()
    for box in boxes:
        tracker = cv2.TrackerKCF_create()
        trackers.add(tracker, frame, tuple(box))

    #Update the tracker for each frame
    success, boxes = trackers.update(frame)
    a=0
    if success:
        # Draw the tracked boxes on the frame
        for box in boxes:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1,p2,colors[a],3)
            if a<len(colors)-1:
                a+=1
            else:
                a=0
            
    else:
    # Remove the tracker if it fails to track the object
        tracker.clear()
    
    num_butterflies = len(trackers.getObjects())
    cv2.putText(frame, f"Number of Butterflies: {num_butterflies}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Indicate when a butterfly flies in or out of frame by drawing a bounding box or arrow
    for i, box in enumerate(boxes):
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if (x < 0)|(x+w > frame_width)|(y<0)|(y+h)>frame_height:
            # Butterfly has flown out of frame on the left side
            cv2.putText(frame, "Butterfly is flying out of frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    result.write(frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
        
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()


# In[ ]:




