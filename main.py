import cv2
import numpy as np 

img_file = "image_demo.jpg"

## Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # network

classes = [] ## our class objects: person, chair, table, ...
with open("coco.names", "r") as f:
	classes =[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

## Load Image
img = cv2.imread(img_file)
# img = cv2.resize(img, None, fx = 0.4, fy = 0.4)

## Keep track of height and width
height, width, channels = img.shape

## Convert Image into Blob
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False) 
## blobFromImage(img, scalefactor, size, mean substraction from each layer, invert Blue with Red, crop)

## Visualize Blob
# for b in blob:
# 	for n, img_blob in enumerate(b):
# 		cv2.imshow(str(n), img_blob)

## Output
net.setInput(blob)
outs = net.forward(output_layers) # forward pass to compute output layer

## Creating array for output name
class_ids = []
confidences = []
boxes = []

## Output in screen
for out in outs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence > 0.5:

			# print("out:", out)
			# print("detection:", detection)
			# print("scores:", scores)
			# print("class_id:", class_id)
			# print("confidence:", confidence)

			center_x = int(detection[0] * width)
			center_y = int(detection[1] * height)
			w = int(detection[2] * width)
			h = int(detection[3] * height)

			## Rectangle box coordinates
			x = int(center_x - w/2) ## top left x
			y = int(center_y - h/2) ## top left y
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

			boxes.append([x , y, w, h])
			confidences.append(float(confidence))
			class_ids.append(class_id)

print("Number of objects detected: ", len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) ## For Non-Maximum Suppression
# print(indexes)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(classes), 3))

for i in range(len(boxes)):
	if i in indexes:
		x, y, w, h = boxes[i]
		label = str(classes[class_ids[i]])
		color = colors[class_ids[i]]
		cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		cv2.putText(img, label, (x, y+h-10), font, 2, color, 2)
		print(label)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()