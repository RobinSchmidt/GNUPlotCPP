#pragma once

// todo: maybe make a separate project for this code, so the mess is not looked at by users

/** Some experimental plots - to try out new stuff. Some things from these functions here may 
eventually be integrated into the class GNUPlotter or to the demos, when they are finished. The 
code here may be messy - it's not supposed to be looked at by users of GNUPlotCPP. */



void curveExperiment2D();
void surfaceExperiment();
void complexExperiment();
void complexCurve();
void complexReIm();
void vectorFieldExperiment();
void zedSquaredVectorField();

// under construction:
void demoVectorField();

void testInitialValueSolver();
void demoComplexDipole();

void testHistogram();
void testMoebiusStrip();

void testSchroedinger();




//=================================================================================================
// utitlity functions and classes to generate data for plotting - maybe eventually move them into
// GNUPlotter.h/cpp or make a dedicated pair of files DataGenerators.h/cpp or maybe 
// PlotUtilities.h/cpp - very preliminary

template<class T>
struct Vector2D
{
  Vector2D(T x_ = 0, T y_ = 0) : x(x_), y(y_) {}
  T x, y;
};
// implement +,- operators, dot- and cross-product (to be written as v.dot(w), v.cross(w))



/** Baseclass for generating all sorts of curves. The template parameter should be a 
vector/point-like type such as Vector2D. Subclasses must implement the actual drawing algorithms, 
for example, based on a parametric equation, implicit equation, differential equation, splines, 
etc. */

template<class TVec>
class CurveGenerator
{

public:

  /** Function that needs to be overriden to produce the curve. It should resize the vector/array 
  as needed and write the points on the curve into it. */
  virtual void getCurve(std::vector<TVec>& curve) = 0; 

};


/*
template<class TScl, class TVec>  // scalar and vector types
class FiledLineGenerator : public CurveGenerator<TVec>
{

public:

  virtual void getCurve(std::vector<TVec>& curve) override;

protected:


  InitialValueSolver<TScl>* stepper = nullptr;

};
*/

