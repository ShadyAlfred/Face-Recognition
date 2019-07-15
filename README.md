# Face Recognition CLI

## Arguments

### -c
Prints the number of faces detected in the image, and create an image file with faces highlighted. Doesn't require internet connection.

### -w
Prints the name of celebrities found in the image. Requires internet connection to use the ClarifaiAPI.

### -i
Prints the predicted age, gender, ethnicity of the face detected. Requires internet connection to use the ClarifaiAPI.


## Packages
* clarifai.rest
* argparse
* cv2
* matplotlib
* socket
* os