#ifndef GNUPLOTTERDEMOS_H
#define GNUPLOTTERDEMOS_H

#include "GNUPlotter.h"


void demoArrayPlot();
void demoFunctionTablePlot();
void demoFunctionPlot();

// void demoFunctionFamily();

void demoAliasing();
void demoFrequencyResponse();

void demoPoleZeroPlotS();
void demoPoleZeroPlotZ();

void demoTransferMagnitude();   // not yet finished

void demoPlottingStyles();
void demoTrigFunctions();
void demoMultiPlot1();
void demoMultiPlot2();

void demoSquare();
void demoMatrixData();

void demoLissajous();
void demoLissajous3D();
void demoHelix();  
void demoPhasor();
void demoLorenz();

void demoTorus(int style = 1);
void demoHenneberg();
void demoGaussianBivariate();
void demoPow();
void demoSincRadial();
void demoSincRadialHeatMap();

void demoPendulumPhasePortrait();

void demoDipole();

void demoContourMap();






// make demos that show various linestyles, pointtypes, etc - provide a legend that shows the
// corrsponding GNUPlot command

#endif
