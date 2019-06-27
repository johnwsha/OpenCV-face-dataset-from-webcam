Using OpenCV to capture still frames from a webcam input for building a facial recognition training dataset. 

To run the program:

python3 build_face_dataset_webcam.py --output dataset/john
--output dataset/["person name here"]

This will create a directory for the person (eg: John) and store the .png picture files there.

Use spacebar to capture images of each person. Try using various head positions for a more diverse training set.

During the recording/saving phase, a green box will show when a face is detected. 

The images will be saved with a dimension of 1280x720 which can later be resized via the "Resizing Images for Training" section of the OpenCV Facial Detection + Recognition git repo.
