#include "Demos.h"
#include "Tests.h"
#include "Experiments.h"

int main(int argc, char** argv)
{
  // This needs to be fixed:
  //demoContourMap();
  // This looks now ugly! Check setToDarkMode and setPixelSize in GNUPlotter. There are
  // versions fo the code marked as "OLD" and "NEW". With the new vesrions, it looks 
  // uglier -> Figure out why and fix it! I think, we need the new versions to make the
  // different terminals work. Check the generated command file for both versions of the 
  // code

  surfaceExperiment();
  surfaceExperiment2();

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
  demoMultiPlot1();
  demoMultiPlot2();    // column spacing is too large!
  demoSquare();
  demoMatrixData();
  demoLissajous();
  demoLissajous3D();   // maybe make a trefoil knot instead
  demoHelix();
  demoPhasor();
  demoLorenz();
  demoTorus();
  demoHenneberg();
  demoGaussianBivariate();
  //demoPow(); // boring - remove
  demoSincRadial();
  demoSincRadialHeatMap();
  demoVectorField();            // is still in Experiments.cpp -> move to Demos.cpp
  demoPendulumPhasePortrait();
  demoDipole();
  demoContourMap();



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
  //surfaceExperiment2();
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
