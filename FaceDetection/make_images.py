import cv2 as cv
import os

# According to the code excerpt, the variable val_ds stands for a dataset object or data structure that contains validation data.


unprocessed_image = 'unprocessed-image'
output_image = 'output-image'

digits = 1

for a_path, bn, fn in os.walk(unprocessed_image):
    for z in fn:
        zp = os.path.join(a_path, z)

        rs = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        for pl in rs:
            igc = cv.imread(zp)

            # With the help of the OpenCV cv.resize function,
            # the image igc is resized in this line to a new size of (180, 180) pixels. While preserving
            # the aspect ratio, it alters the image's dimensions.
            igc = cv.resize(igc, (180, 180))

            # Utilizing the cv.cvtColor function, this line changes the image's igc from grayscale to the BGR color space.
            # The image is converted to a single-channel grayscale representation by stripping away the color information.
            igc = cv.cvtColor(igc, cv.COLOR_BGR2GRAY)
            ui, ci = igc.shape
            M = cv.getRotationMatrix2D(((ci - 1) / 2.0, (ui - 1) / 2.0), pl, 1)

            # In this line, the cv.warpAffine function is used to transform the image igc using the affine
            # transformation provided by the rotation matrix M. On the basis of the determined transformation matrix,
            # it rotates the image actually.
            igc = cv.warpAffine(igc, M, (ci, ui))

            rdi = os.path.relpath(a_path, unprocessed_image)

            # This line of code accomplishes two things. The relative subdirectory path is first represented by concatenating the output_image
            # directory path with rdi. The variable osub holds the resulting path. The directory given by osub is then created, making sure it
            # already exists. No errors are raised if the directory already exists, and it is created, together with any necessary parent directories,
            # if it doesn't.
            osub = os.path.join(output_image, rdi)
            os.makedirs(osub, exist_ok=True)

            ofna = str(digits) + '.jpeg'

            output_path = os.path.join(osub, ofna)

            # The two goals of this code fragment are achieved. The image igc is first written using cv.imwrite() to the
            # file system at the specified output_path. In order to retain the processed photos' sequential order and guarantee
            # unique filenames, it also increases the digits counter by 1, most likely by 1.

            cv.imwrite(output_path, igc)
            digits = digits + 1
