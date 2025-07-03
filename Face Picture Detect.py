import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Read the input image
input_image = cv2.imread(r"C:\Users\Ephraim Oche Eche\PycharmProjects\pythonProject6\Whatapp Picture.jpg")

# Convert the image to grayscale (required for face detection)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

# Draw green rectangles around the detected faces
for (x, y, width, height) in detected_faces:
    cv2.rectangle(input_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green color (BGR format)

# Display the output image with detected faces
cv2.imshow("Detected Faces", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()