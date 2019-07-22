/* Author: Eric C Joyce
           Stevens Institute of Technology

   A small, stand-alone program that facilitates the Python script, point.py. 
   This program opens the given image in a clickable window and returns the image
   coordinates of every mouse click it receives. Green circles appear over clicks.
   Press any key to close the window.

   Compile: g++ clicker.cpp -o clicker `pkg_config --cflags --libs opencv`
   Usage: ./clicker imagefilename
*/
#include <opencv2/imgproc.hpp>                                      //  For annotating our GUI
#include <opencv2/highgui.hpp>                                      //  For the display window
#include <iostream>

/*
#define __CLICKER_DEBUG 1
*/

using namespace std;
using namespace cv;

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Prototypes  */

void onMouseClick(int, int, int, int, void*);                       //  Mouse-click callback function
void drawReferenceCircle(Mat&, Point2i);                            //  Show where we've placed a reference

/**************************************************************************************************
 Functions  */

/* Respond to mouse clicks and update reference points */
void onMouseClick(int event, int x, int y, int flags, void* param)
  {
    Mat& img = *(Mat*)param;                                        //  Recover the working-copy image
    Point2i p = Point2i(x, y);                                      //  Pack mouse-click into a cv::Point

    if(event == EVENT_LBUTTONDOWN)                                  //  Left-click
      {
        cout << x << "," << y << endl;
        drawReferenceCircle(img, p);                                //  Show where the user set a point
      }

    return;
  }

/* Put a green circle over the given point
   to indicate that a reference point has been established. */
void drawReferenceCircle(Mat& img, Point2i p)
  {
    circle(img, p, 3, Scalar(0x00, 0xff, 0x00), 2, 8, 0);
    return;
  }

int main(int argc, char** argv)
  {
    Mat src;                                                        //  Original source image

    if(argc < 2)                                                    //  We require that the user specify an image
      {
        cout << "Please provide an image file." << endl;
        cout << "Usage: ./clicker image-file-name" << endl;
        return 1;
      }

    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);                     //  Read the image file indicated
    if(!src.data)                                                   //  Broken?
      {
        cout << "Error: could not open or find the image \"" << argv[1] << "\"." << endl;
        return 1;
      }

    namedWindow("Click Me", WINDOW_AUTOSIZE);                       //  Build a window containig the image
                                                                    //  Attach a mouse callback function
    setMouseCallback("Click Me", onMouseClick, (void*)&src);

    for(;;)
      {
        imshow("Click Me", src);                                    //  Show/Refresh the window
        if(waitKey(15) != -1)                                       //  Loop until user hits a key
          break;
      }

    destroyWindow("Click Me");                                      //  Finally tear down the info-collection window

    return 0;
  }
