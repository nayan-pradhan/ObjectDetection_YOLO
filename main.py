import cv2
import numpy as np 

## Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # network

classes = [] ## our class objects: person, chair, table, ...
with open("coco.names", "r") as f:
	classes =[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

## Load Image
img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx = 0.4, fy = 0.4)

## Convert Image into Blob
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False) 
## blobFromImage(img, scalefactor, size, mean substraction from each layer, invert Blue with Red, crop)

## Visualize Blob
# for b in blob:
# 	for n, img_blob in enumerate(b):
# 		cv2.imshow(str(n), img_blob)

## Output
net.setInput(blob)
out = net.forward(output_layers) # forward pass to compute output layer

# ## Output in screen
# for out in outs:
# 	for detection in out:
		


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


