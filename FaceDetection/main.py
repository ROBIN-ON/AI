import cv2 as cv
import tensorflow as tf
import numpy as np
import datetime

# Using TensorFlow's load_model function, this snippet of code loads a pre-trained model that is located at the file path "model/my_model."

model = tf.keras.models.load_model('model/my_model')

# A new sequential model, probability_model, is then made using the loaded model by adding the loaded model as the first layer and adding
# a softmax layer afterward. By doing this, the probability_model is able to take input data, run it through the loaded model, and then use
# the model's output to provide probabilistic predictions by applying the softmax activation function.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# Using the following code snippet, you can make a list named class_names that has a number of names that correspond
# to various classes. Each class name is viewed in this context as a separate string element within the list.
# As a result, the code essentially creates a mechanism for class names to be conveniently organized and stored,
# enabling their use or manipulation within a program or system.

class_names = ['dawa', 'kripa' , 'kritika', 'prasanna', 'robin']



# Constants for the green box are specified to guarantee consistent values, assisting in the maintenance and clarity of the code.
BOX_SIZE = 350
BOX_COLOR = (0, 255, 0)  # green color in BGR format

# Making a function especially made to handle important events
# will ensure effective handling and responsiveness across the codebase.
def key_event_handler(key):
    if key == 32:  # Space key
        global captured_image
        output_image = preprocess_image(captured_image)
        prediction = predict_with_model(output_image)
        print("Prediction:", class_names[np.argmax(prediction[0])])
        f = open('presence.txt', 'a')
        f.write(class_names[np.argmax(prediction[0])] + ' ' + str(datetime.date.today()) + '\n' )
        f.close()

# In order to optimize the input format and increase compatibility for effective processing,
#     a specialized function is built to preprocess the image in advance of the TensorFlow model.


def preprocess_image(image):
    # The image is cropped to the required input size, making it compatible with
    # the intended standards and permitting easy integration with future procedures or algorithms.
    resized_image = cv.resize(image, (180, 180))

    # The data is normalized, assuring consistency in scale for further computations or analysis,
    # by adjusting the pixel values to fit within the range of [0, 1].
    normalized_image = resized_image / 255.0

    # Assuming a 3-channel image format, the image's dimensions are increased to fit the input
    # geometry of the model. This modification provides compatibility and accurate alignment with
    # the anticipated input structure of the model.
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# The TensorFlow model is used to execute predictions using a specific function,
# allowing for effective inference and offering a simplified method for utilizing the model's capabilities.
def predict_with_model(image):

    # refers to the method of using a trained machine learning model to produce an output or estimate from provided
    # input data. It entails using the learnt patterns and correlations of the model to apply to fresh or previously
    # unobserved data and derive a prediction or inference.

    prediction = probability_model.predict(image)
    return prediction

# Launch the camera that has been set as the default device, enabling it to work and getting
# it ready to take pictures or videos in accordance with the system's default settings.
camera = cv.VideoCapture(0)

# it helps to check and monitor if the camera is opened properly or not which obviously helps to detect the face
if not camera.isOpened():
    print("Failed to open camera")
    exit()

captured_image = None

while True:
    # Obtain the most recent image or video frame that was acquired by the
    # camera device by requesting it from the camera, then processing or analyzing it.
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Get the width and height values for the frame, which determine its size or resolution.
    # This information is crucial for any further actions or calculations using the frame.
    height, width, _ = frame.shape

    # Calculations that provide the precise location in the box's center can be used to determine the central coordinates
    # of the green box. This information is useful for numerous activities like object tracking and alignment.
    x = int((width - BOX_SIZE) / 2)
    y = int((height - BOX_SIZE) / 2)

    # call the green box's outline or border onto the frame to represent it visually. This will improve the visual output and call
    # attention to the green box's presence or position within the frame of the video or image that was shot.
    cv.rectangle(frame, (x, y), (x+BOX_SIZE, y+BOX_SIZE), BOX_COLOR, 2)

    # Display the frame on a visual display, displaying the still or moving image or video frame on a screen or monitor for human analysis
    # or observation, and offering a real-time or static representation of the acquired content for visual examination or further processing.
    cv.imshow("Camera", frame)

    # It is possible to respond quickly and dynamically to user commands or instructions within a software or system by monitoring and examining
    # the occurrence of key events, identifying, and analyzing any keyboard interactions that may have occurred.
    key = cv.waitKey(1) & 0xFF
    if key != 255:
        key_event_handler(key)

    # Set the green box's defined area as the only area to be included in the image capture process.
    # This will isolate and highlight the desired region of interest.
    captured_image = frame[y:y+BOX_SIZE, x:x+BOX_SIZE].copy()

    # If the "q" key is pushed, the loop's execution will end and the program will immediately
    # stop its current iteration or repetitive sequence of instructions,
    # allowing it to halt and stop further processing in response to user input.
    if key == ord('q'):
        break

# In order to properly release and dispose of any allocated camera-related objects or connections and to effectively stop the camera's
# functionality and visual display for the user, free the camera's resources and shut all open windows linked to the camera feed.
camera.release()
cv.destroyAllWindows()