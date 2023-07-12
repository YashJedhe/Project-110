# To Capture Frame
import cv2

# To process image array
import numpy as np
import tensorflow as tf

mymodel=tf.keras.models.load_model("keras_model.h5")


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		img=cv2.resize(frame,(224,224))

		# expand the dimensions
		resized_frame = np.expand_dims(resized_frame , axis = 0)
		
		# normalize it before feeding to the model
		resized=resized_frame/255

		# get predictions from the model
		prediction=mymodel.predict(resized_frame)
		
                #Conerting the data in the array to precentage confidence
		rock = int(prediction[0][0]*100)
		paper = int(prediction[0][1]*100)
		scissor = int(prediction[0][2]*100)

		#printing percentage confidence
		print(f"Rock: {rock}%, Paper: {paper}%, Scissor: {scissor}%")
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
