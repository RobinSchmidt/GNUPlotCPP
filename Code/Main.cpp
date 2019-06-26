#include "Demos.h"
#include "Tests.h"
#include "Experiments.h"

int main(int argc, char** argv)
{
  //testMatrixWrite();
  //testLowLevelCalls();

  
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
  demoSquare();
  demoMatrixData();
  demoLissajous();
  demoLissajous3D();   // maybe make a trefoil knot instead
  demoHelix();
  //demoPhasor();
  demoTorus();
  demoHenneberg();
  demoGaussianBivariate();
  //demoPow(); // boring - remove
  demoSincRadial();
  demoSincRadialHeatMap();
  demoPendulumPhasePortrait();
  demoDipole();

  // If you want to see, how they were done, jump into the implementations of these functions - 
  // there you will find commented code that should make it clear how the plotter class is supposed
  // to be used , so you can take these as examples and reference for using it for creating your 
  // own plots.
  



  // experimental plots - not yet finsihed:

  //testInitialValueSolver();
  //testLorenz();

  //demoVectorField();
  //demoDipole();  // not so interesting - we should draw field lines, not the vector field

  // these are just some tests for debugging:
  //testDataSetup();
  //testDataSetup2();

  // Experiments:
  //curveExperiment2D();
  //surfaceExperiment();
  //complexExperiment();
  //vectorFieldExperiment();
  //zedSquaredVectorField();


  getchar();
  return(EXIT_SUCCESS);
}
