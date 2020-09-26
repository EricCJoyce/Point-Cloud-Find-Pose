# Point-Cloud-Find-Pose
Estimate the position in a point cloud from which a photo was taken. This script uses user clicks to establish correspondences.

## Dependencies and Set-Up

### NumPy
http://www.numpy.org/
Needed for vector and matrix operations, including matrix inversion
### OpenCV
https://opencv.org/
Needed for image manipulation, feature detection, and the PnP-solver. 

This script calls upon a stand-alone program named `clicker`, the C++ code for which is in `clicker.cpp`. This means you will need to be able to compile programs that use the OpenCV library, which is a different process than Python's `import cv2`. The following worked for us, installing OpenCV 3.1.0 on Ubuntu 16.04:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install git libgtk-3-dev
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev
sudo apt-get install libtbb2 libtbb-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev sphinx-common yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev
sudo apt-get install libatlas-base-dev gfortran
```
The foregone commands install all the requisite libraries. Now we install and build OpenCV:
```
sudo -s
cd /opt
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip
mv /opt/opencv-3.1.0/ /opt/opencv/
mv /opt/opencv_contrib-3.1.0/ /opt/opencv_contrib/
cd opencv
mkdir release
cd release
cmake -D WITH_IPP=ON -D INSTALL_CREATE_DISTRIB=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/
make
make install
ldconfig
exit
cd ~
```
If all went well, we should be able to query the version of the OpenCV installation:
```
pkg-config --modversion opencv
```
We should also be able to compile C++ code that uses OpenCV utilities. (Suppose we've written some such program, `helloworld.cpp`.)
```
g++ helloworld.cpp -o helloworld `pkg-config --cflags --libs opencv`
```
### Matplotlib
https://matplotlib.org/
Needed for rendering arrays to images.
### Open3D
http://www.open3d.org/
Needed for point-cloud visualization and depth-buffer rendering.
### face.py
The classes in this file assist the `find.py` script. It should be in the same directory.

## Inputs

### Example Script Call
The only required arguments are an image and a point-cloud file, as seen here: `python point.py image.jpg pointcloud.ply`
Script performance can be modified by passing flags and arguments after the point cloud. Please see these described below in "Parameters."

## Outputs

(More on this later)

## Parameters

### `-K` Specify a Camera Matrix
Some mathematical information about the camera that took the picture given to the algorithm is necessary to form an estimate. By default, the script looks for `K.mat`, which is really just a place-holder name. Whatever file you give to `-K` should be an ASCII text file with the following format:
* Required parameters are `fx`, `fy`, `cx`, `cy`, `w`, and `h`.
* Parameters each begin a single line and are separated with a tab from their values.
* Comment lines are ignored. They begin with the `#` character
* `fx` is the horizontal focal length in pixels
* `fy` is the vertical focal length in pixels
* `cx` is the X of camera's principal point (usually equal to half the image width) in pixels
* `cy` is the Y of camera's principal point (usually equal to half the image height) in pixels
* `w` is the image width in pixels
* `h` is the image height in pixels

These are the main mathematical details of a camera needed to make inferences about where observed points lie (we ignore such exotic factors as lens distortion). If you know your camera's model, look for its technical details here: https://www.digicamdb.com/. Our experiments used photos taken by an iPhone XS. Unfortunately iPhone cameras are not listed in the Database, but enough details were found in online tech articles to make a good estimate of the XS matrix. The result is the file `Apple-iPhoneXS-P4.2mm.mat` in this repository. Anticipate that you will have several `.mat` files for all the cameras you work with. The numbers in this script's equations change with the camera's focal length, and if the image given to it uses a landscape or a portrait composition. It was helpful to have a reminder of this; hence the "PORTRAIT" comment and the "P" in the file name `Apple-iPhoneXS-P4.2mm.mat`. Here is an example script call using `-K`: `python point.py image.jpg pointcloud.ply -K Apple-iPhoneXS-P4.2mm.mat`

### `-v` Enable Verbosity
It is often helpful for the script to exhibit signs of life. Enabling verbosity lets a user know what's going on. For example: `python point.py image.jpg pointcloud.obj -v`

### `-showFeatures` Generate a 3D PLY File for All Points
Enabling `-showFeatures` tells the script to create a new 3D point-cloud (PLY) file once feature-correspondences have been established. The file `features.ply` will contain as many three-dimensional points, colored bright green, as there are 2D-to-3D correspondences. `python point.py image.jpg pointcloud.ply -showFeatures`

### `-o` Add an Output Format
Several or no output formats can be specified on the command line. As mentioned above, when no output formats are given, the script prints its results to the screen as a pair of 3-vectors. The following output formats are recognized:
* `rt` Write the rotation and translation vectors to file
* `Rt` Write the rotation matrix and translation vector to file
* `P` Compute a projection matrix from the rotation and translation vectors and write that matrix to file
* `Meshlab` Write copy-pastable Meshlab markup to file. (The formatting of this markup may differ according to your version of Meshlab. It should be mentioned that I have not gotten this feature to work yet: https://sourceforge.net/p/meshlab/discussion/499533/thread/cc40efe0/)
* `Blender` Write copy-pastable Blender Python console code to file.

Any and all output formats are written to the same file, `point.log`. For example: `python point.py image.jpg pointcloud.ply -o Rt -o P -o Blender`

### `-?`, `-help`, `--help` Help!
Display some notes on how to use this script.

## Citation

If this code was helpful for your research, please consider citing this repository.

```
@misc{point-cloud-find-pose_2019,
  title={Point-Cloud-Find-Pose},
  author={Eric C. Joyce},
  year={2019},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/EricCJoyce/Point-Cloud-Find-Pose}}
}
```
