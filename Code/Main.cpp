#include "Demos.h"
#include "Tests.h"
#include "Experiments.h"

// Notes/ToDo:
//
// - Many of the plots look wrong when using Windows's scaling feature (found in  
//   Settings -> System -> Display -> Scale in Win 11). When the scale is set to 100%, the plots 
//   look fine. 100% is appropriate for a normal Full-HD screen with a ressolution of 1920x1080 
//   pixels. On an UHD screen with with 3840x2160 pixels, I usually use a scaling of 200%. But 
//   that makes some of the plots look wrong. Legends become too big, ticks too dense, etc. I'm not
//   sure, how to deal with that. Maybe we can somehow figure out the current screen resolution and
//   scaling and use setPixelSize() with values that are most appropriate for the current scaling
//   setting. I think, we just should multiply the base pixel sizes by the scaling factor. Maybe
//   the function setPixelSize() should take an optional boolean parameter "useSystemScaling"
//   defaulting to false (old behavior).
//
// - Check what's going on with the demoMultiPlot1/2 functions. They behave differently from the
//   others (see comments). Also demoContourMap() uses this other GUI. And it looks ugly. It used
//   to look much better on my old PC.

int main(int argc, char** argv)
{
  // This needs to be fixed:
  //demoContourMap();
  // This looks now ugly! Check setToDarkMode and setPixelSize in GNUPlotter. There are
  // versions fo the code marked as "OLD" and "NEW". With the new vesrions, it looks 
  // uglier -> Figure out why and fix it! I think, we need the new versions to make the
  // different terminals work. Check the generated command file for both versions of the 
  // code

  // Here are a couple of functions that demonstrate the use of the plotter. Whenever you close the
  // GNUPlot window, the currently running demo function will return and the next demo is launched:
  demoArrayPlot();
  demoFunctionTablePlot();
  demoFunctionPlot();
  demoAliasing();
  demoFrequencyResponse();
  demoPoleZeroPlotS();
  demoPoleZeroPlotZ();
  demoPlottingStyles();
  demoTrigFunctions();
  demoMultiPlot1();    // Opens a different GUI then the others. Does not wait for closing.
  demoMultiPlot2();    // Dito. Also: Column spacing between subplots is too large! 
  demoSquare();
  demoMatrixData();
  demoLissajous();
  demoLissajous3D();   // maybe make a trefoil knot instead
  demoHelix();
  demoPhasor();
  demoLorenz();
  demoTorus(0);
  demoTorus(1);
  demoTorus(2);
  demoHenneberg();
  demoGaussianBivariate();
  //demoPow(); // boring - remove
  demoSincRadial();
  demoSincRadialHeatMap();
  demoVectorField();            // Is still in Experiments.cpp -> move to Demos.cpp
  demoPendulumPhasePortrait();
  demoDipole();
  demoContourMap();             // This looks ugly! Also uses the different GUI



  // If you want to see, how they were done, jump into the implementations of these functions - 
  // there you will find commented code that should make it clear how the plotter class is supposed
  // to be used, so you can take these as examples and reference for using it for creating your 
  // own plots.
  
  // Experimental plots - not yet finished:

  //testInitialValueSolver();
  //contours();

  // these are just some tests for debugging:
  //testDataSetup();
  //testDataSetup2();

  // Experiments:
  //curveExperiment2D();
  //surfaceExperiment();
  //complexExperiment();
  //complexCurve();
  //complexReIm();
  //vectorFieldExperiment();
  //zedSquaredVectorField();

  //testHistogram();
  //testMoebiusStrip();
  //testSchroedinger();
  //testDrawing();
  //testRotation();
  //testAnimation();        // produces a .gif file in the project directory, doesn't open a window
  

  printf("That's it! All demo plots have been shown.");
  getchar();
  return(EXIT_SUCCESS);
}
