import cv2 as cv

#! THESE ARE IMAGES THAT AREN'T DOWNSIZED
#original_image_1 = cv.imread("hamburger_face.JPG")
#original_image_2 = cv.imread("hammock_reading.JPG")
#original_image_3 = cv.imread("sofa_face.JPG")
#original_image_4 = cv.imread("frisbee_team.JPG")
original_image_5 = cv.imread("mans_face.JPG")

# TO PRINT OUT ARRAY AND DIMENSIONS
# print(original_image)
# print(original_image.shape)

#grayscale_image = cv.cvtColor(original_image_1, cv.COLOR_BGR2GRAY)
#grayscale_image = cv.cvtColor(original_image_2, cv.COLOR_BGR2GRAY)
#grayscale_image = cv.cvtColor(original_image_3, cv.COLOR_BGR2GRAY)
#grayscale_image = cv.cvtColor(original_image_4, cv.COLOR_BGR2GRAY)
grayscale_image = cv.cvtColor(original_image_5, cv.COLOR_BGR2GRAY)

# TO PRINT OUT GRAYSCALE IMG
#cv.imshow("gray_img", grayscale_image)
#cv.waitKey(0)
#cv.destroyAllWindows()

face_cascade = cv.CascadeClassifier('haar_cascade_front.xml')
detected_faces = face_cascade.detectMultiScale(grayscale_image)

# PRINTS COORDINATES OF FACES
#print(detected_faces)

for face in detected_faces:
    x , y , w , h = face
    cv.rectangle(original_image_5, (x, y), (x + w , y + h ), (0 , 255 , 0), 2)

cv.imshow("orig_img", original_image_5)
cv.waitKey(0)
cv.destroyAllWindows()