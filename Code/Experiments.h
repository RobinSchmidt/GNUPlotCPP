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


void testInitialValueSolver();
void testLorenz();

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

  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  /** This sets up the actual derivative computation function that defines your system of ordinary
  differential equations. It should have two arguments which are pointers to arrays of a size
  given by numDimensions. The first array is the input vector and the second is the output. The
  two pointers may point to the same memory location - so the state update can be done in place. 
  We treat non-autonomous systems uniformly with autonomous ones - you just add the identity 
  function as first element to the function vector. So, if you have a non-autonomous system (i.e. 
  your vector valued derivative computation function has an explicit time dependency), you should 
  add one dimension and your derivative computation function should write the constant 1 into the 
  first slot of the derivative vector at each step. This nicely generalizes to ODE systems with 
  more than one independent variables. */
  virtual void setDerivativeFunction(const std::function<void(const T*, T*)>& function, 
    int numDimensions)
  {
    deriv = function;
    y.resize(numDimensions);
    yPrime.resize(numDimensions);
    k1.resize(numDimensions); 
    k2.resize(numDimensions); 
    //tmp.resize(numDimensions);
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

  //-----------------------------------------------------------------------------------------------
  /** \name Processing */

  /** Performs a forward Euler step: yOut = yIn + stepSize * yPrime. yOut may point to the same 
  array as yIn for in-place update. This is the simplest numerical integration scheme for ODEs. */
  virtual void stepEuler(const T* yIn, T* yOut)
  {
    deriv(yIn, &yPrime[0]);
    for(size_t i = 0; i < yPrime.size(); i++)
      yOut[i] = yIn[i] + h * yPrime[i];
  }

  /** Under construction - needs test, if it is really more accurate - ok - this seems to work */
  void stepMidpoint(const T* yIn, T* yOut)
  {
    deriv(yIn, &yPrime[0]);    // compute y'[yIn]

    for(size_t i = 0; i < yPrime.size(); i++) {
      k1[i] = h * yPrime[i];
      k2[i] = yIn[i] + 0.5 * k1[i]; // fill k2 with evaluation at the half-step
    }

    deriv(&k2[0], &yPrime[0]); // compute y' at midpoint yIn + (h/2) * y'[at yIn]

    // do full step with y' computed at midpoint:
    for(size_t i = 0; i < yPrime.size(); i++)
      yOut[i] = yIn[i] + h * yPrime[i];
  }

  /** Under construction */
  void stepMidpointWithError(const T* yIn, T* yOut, T* error)
  {
    // write step result into temporary buffer, because we still need yIn, which would be 
    // overwritten in in-place usage (i.e. if yIn == yOut):
    stepMidpoint(yIn, &y[0]);

    // estimate error (this is where we still need yIn) and then write midpoint step result into
    // output:
    for(size_t i = 0; i < y.size(); i++) {
      error[i]  = y[i] - (yIn[i] + k1[i]);  // error-estimate = midpoint-result - Euler-result
      yOut[i] = y[i];                       // ..(k1 still contains the Euler step)
    }


    // the error estimate is pessimistic - we actually estimate the error of the Euler step (by
    // comparing it against the midpoint step, which is actually taken)
  }

  /** Needs test */
  void stepMidpointAndAdaptSize(const T* yIn, T* yOut, T* error = nullptr)
  {
    // repeat trial midpoint steps until the result is within the desired accuracy:
    while(true) { // todo: implement a safeguard against infinite loops
      stepMidpointWithError(yIn, &y[0], &err[0]);
      if(isAccurateEnough(&err[0]))
        break;    // result is good enough - we accept this step
      else
        h *= 0.5; // decrease step-size and try again (factor 0.5 is chosen ad-hoc)
    }

    // increase step-size for *next* step, if appropriate:
    if(isTooAccurate(&err[0]))
      h *= 2.0;   // factor 2 ad-hoc

    // copy step result into output:
    for(size_t i = 0; i < y.size(); i++)
      yOut[i] = y[i];

    // copy error estimate into error output, if caller is interested in it:
    if(error != nullptr)
      for(size_t i = 0; i < err.size(); i++)
        error[i] = err[i];
  }

  bool isAccurateEnough(const T* error)
  {
    for(size_t i = 0; i < y.size(); i++)
      if( abs(error[i]) > accuracy )
        return false;
    return true;
  }

  bool isTooAccurate(const T* error)
  {
    for(size_t i = 0; i < y.size(); i++)
      if( abs(error[i]) < 0.5 * accuracy ) // factor 0.5 selected ad-hoc - maybe this has to be 
        return false;                      // modified - see what NR says about this and make tests
    return true;
  }



  /** Not yet finished */
  virtual void stepRungeKutta(const T* yIn, T* yOut)
  {
    deriv(yIn, &yPrime[0]);
    // ...
  }




  /** Not yet finished */
  virtual T stepEulerWithError(const T* yIn, T* yOut)
  {
    stepEuler(yIn, yOut);
    return T(0);
    // preliminary - todo: do an Euler step and a (possibly embedded) 2nd order step and use their 
    // difference as error estimate
  }

  /** Not yet finished */
  virtual void stepEulerAndAdaptStep(const T* yIn, T* yOut, T* error = nullptr)
  {
    //stepEulerWithError(yIn, &tmp[0], ..)
  }


  // todo:
  // -implement simple RK4 (without stepsize control)
  // -implement error estimation and stepsize control for Euler method
  // -implement RK4 method with embedded 5th order method for error estimation



protected:

  T h = 0.01;                               // integration step size
  std::vector<T> yPrime;                    // to hold dy/dt (vector valued)
  std::function<void(const T*, T*)> deriv;  // function to compute the derivative
  // 1st argument: current state vector (input)
  // 2nd argument: derivative at current state vector (output)
  // client code must set up this function - this function is what determines the actual system of
  // differential equations

  std::vector<T> k1, k2, y; //, err;

  // stuff for stepsize control (factor out):
  T accuracy = 0.001;  // desired accuracy - determines step-sizes - todo: have an array - allow
                       // different accuracies for different variables

  bool stepAdapt = true; // may not be needed
  T hMin = 0.0;
  T hMax = std::numeric_limits<T>::infinity();
  //std::vector<T> tmp;
  // maybe factor out the step-size adaption - have a baseclass with fixed step-size
};


template<class TScl, class TVec>  // scalar and vector types
class FiledLineGenerator : public CurveGenerator<TVec>
{

public:

  virtual void getCurve(std::vector<TVec>& curve) override;

protected:


  InitialValueSolver<TScl>* stepper = nullptr;

};

// other solvers: BoundaryValueSolver, ImplicitEquationSolver (can be used fo equipotentials), 
// in 3D: a solver that finds the intersection curve(s) of two (parametric or implicit) surfaces