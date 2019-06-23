#pragma once

// todo: maybe make a separate project for this code, so the mess is not looked at by users

/** Some experimental plots - to try out new stuff. Some things from these functions here may 
eventually be integrated into the class GNUPlotter or to the demos, when they are finished. The 
code here may be messy - it's not supposed to be looked at by users of GNUPlotCPP. */








void curveExperiment2D();
void surfaceExperiment();
void complexExperiment();
void vectorFieldExperiment();
void zedSquaredVectorField();

// under construction:
void demoVectorField();
void demoComplexDipole();

void testDipole();



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



/** Class for solving an initial value problem for a system of first-order ordinary differential 
equations. */

template<class T>  // T is a scalar type (double or float)
class InitialValueSolver  // maybe rename to InitialValueStepper
{

public:


  /** Sets the number of dimensions that the ODE system has.  We treat non-autonomous systems 
  uniformly with autonomous ones - you just add the identity function as first element to the 
  function vector. So, if you have a non-autonomous system (i.e. your vector valued derivative 
  computation function has an explicit time dependency), you should add one dimension and your 
  derivative computation function should write the constant 1 into the first slot of the derivative
  vector at each step. */
  virtual void setNumDimensions(int newNumber)
  {
    numDimensions = newNumber;
  }

  virtual void init(const std::vector<T>& initialState, T initialTime = 0) 
  { 
    x = initialState;
    t = initialTime;
  }



  /** Sets the size of steps to be taken. Not that if step size adaption is used, the value here
  will only be used as initial value and may change over time. If you want to use the given step 
  size as fixed step size, call setStepSizeAdaption(false). */
  virtual void setStepSize(T newSize)
  {
    h = newSize;
  }

  /** Switches step size adaption on/off.  */
  virtual void setStepSizeAdaption(bool shouldAdapt)
  {
    stepAdapt = shouldAdapt;
  }

  /** Sets the desired accuracy. Relevant only, when adaptive step size control is used. The step 
  size will be updated on the fly according to an error estimate. */
  virtual void setAccuracy(T newAccuracy)
  {
    accuracy = newAccuracy;
  }

protected:

  std::vector<T> x;    // current state


  T accuracy = 0.001;  // desired accuracy - determines step-sizes

  bool stepAdapt = true;

  int numDimensions = 1;

  // value and range for the "time" parameter:
  T t    = 0;
  //T tMin = 0.0;
  //T tMax = 1.0;

  // value and range for the time-step:
  T h    = 0.01;
  T hMin = 0.0;
  T hMax = std::numeric_limits<TScl>::infinity();

  //int numDimensions

  int numSteps = 0;    // number of steps taken

  std::vector<T> tmp;  // to hold dx/dt (vector valued)


  std::function<void(const T*, T*)> deriv; 
  // function to compute the derivative: 
  // 1st argument: current state vector (input)
  // 2nd argument: derivative at current state vector (output)
  // client code must set up this function - this function is what determines the actual system of
  // differential equations

  //T minStepSize = 0.0;
  //T maxStepSize = std::numeric_limits<TScl>::infinity();

  // maybe factor out the step-size adaption - have a baseclass with fixed step-size

};


template<class TScl, class TVec>  // scalar and vector types
class FiledLineSolver : public CurveGenerator<TVec>
{

public:

  virtual void getCurve(std::vector<TVec>& curve) override;

protected:


  InitialValueSolver<TScl>* stepper = nullptr;

};

// other solvers: BoundaryValueSolver, ImplicitEquationSolver (can be used fo equipotentials), 
// in 3D: a solver that finds the intersection curve(s) of two (parametric or implicit) surfaces