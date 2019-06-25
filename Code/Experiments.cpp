#include "GNUPlotter.h"
#include "Experiments.h"
//#include <math.h>
using namespace std;

#define M_PI 3.14159265358979323846

//std::vector<double>

double maxError(const std::vector<double>& x, const std::vector<double>& y)
{
  // todo: assert that x and y have the same size
  double maxErr = 0.0;
  for(size_t i = 0; i < x.size(); i++)
    if( fabs(x[i] - y[i]) > maxErr )
      maxErr = fabs(x[i] - y[i]);
  return maxErr;
}

std::vector<double> diff(const std::vector<double>& x)
{
  std::vector<double> d(x.size());
  for(int i = 0; i < x.size()-1; i++)
    d[i] = x[i+1] - x[i];
  d[d.size()-1] = 0;
  return d;
}

template<class T>
T findRoot(const std::function<T(T)>& f, T xL, T xR, T y = T(0))
{
  static const int maxNumIterations = 60; // should be enough for double-precision
  T tol = std::numeric_limits<T>::epsilon();
  T fL  = f(xL) - y;
  T xM, fM;
  for(int i = 1; i <= maxNumIterations; i++) {
    xM = T(0.5)*(xL+xR);
    fM = f(xM) - y;
    if(fM == 0 || xR-xL <= fabs(xM*tol))
      return xM;
    if(fL*fM > 0) { xL = xM; fL = fM; }
    else          { xR = xM;          }
  }
  //rsError("findRoot failed to converge");
  return xM;
}


//-------------------------------------------------------------------------------------------------
// convenience functions for certain types of plots (eventually move to class GNUPlotter):

template<class T>
void plotComplexSurface(const function<complex<T>(complex<T>)>& f,
  int Nr, T rMin, T rMax, int Ni, T iMin, T iMax)
{
  std::function<T(T, T)> fx, fy, fz;
  fx = [&] (T re, T im) { return real(f(complex<T>(re, im))); };
  fy = [&] (T re, T im) { return imag(f(complex<T>(re, im))); };

  // maybe let the user select, what should be used by a parameter:
  //fz = [&] (T re, T im) { return 0; };  // preliminary

  fz = [&] (T re, T im) { return re; };  // preliminary
  // for f(z) = z^2, this looks a lot like the Riemannian surface here:
  // https://www.youtube.com/watch?v=4MmSZrAlqKc&list=PLiaHhY2iBX9g6KIvZ_703G3KJXapKkNaF&index=13
  // ..but is it really the same thing?

  //fz = [&] (T re, T im) { return abs(f(complex<T>(re, im))); };
  //fz = [&] (T re, T im) { return abs(complex<T>(re, im)); };

  //fz = [&] (T re, T im) { return im; };  // preliminary

  GNUPlotter::plotSurface(fx, fy, fz, Nr, rMin, rMax, Ni, iMin, iMax);
}
// maybe try showing abs and arg instead of re and im, also try to use abs and arg as inputs
// ...there seem to be a lot of combinations that may be tried
// ...maybe this should be a class ComplexSurfacePlotter - for a function, the number of 
// parameters would be overwhelming

template<class T>
void plotComplexVectorField(const function<complex<T>(complex<T>)>& f,
  int Nr, T rMin, T rMax, int Ni, T iMin, T iMax)
{
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, false);
}
// maybe have a function plotComplexVectorFieldPolar? does that make sense? or maybe we should 
// generally allow for vector fields where the input is in polar coordinates? that may make sense
// in some cases, for example for a rotationally symmetric gravitational field
// -> plotVectorFieldPolar ...maybe factor out a function that creates the pairs of xy-values

// seems like sucha plot is called Polya-plot - see here
// http://mathworld.wolfram.com/PolyaPlot.html
// here at the bottom, they say something about Polya plots having an additional complex 
// conjugation involved - maybe let the complex conjugation be optional and on by default
// https://mathematica.stackexchange.com/questions/4244/visualizing-a-complex-vector-field-near-poles
// iirc, Visual Complex Analysis says soemthing about this - look up




//-------------------------------------------------------------------------------------------------
// actual experiments:

void curveExperiment2D()
{
  // we plot a Lissajous curve...
  double wx = 3, wy = 2;  // frequencies for x and y functions
  double ax = 1, ay = 1;  // amplitudes

  double tMin = 0; double tMax = 2*M_PI; int Nt = 200;

  std::function<double(double)> fx, fy;
  fx = [&] (double t) { return ax * cos(wx*t); };
  fy = [&] (double t) { return ay * sin(wy*t); };

  GNUPlotter::plotCurve2D(fx, fy, Nt, tMin, tMax);
}

void surfaceExperiment()
{
  // We plot a surface that is defined by 3 std::function objects for x(u,v), y(u,v), z(u,v)

  // Set up range and umber of sampling points for the two paremeters u and v for our parameteric 
  // surface:
  double uMin = -1; double uMax = +1; int Nu = 21;
  double vMin = -1; double vMax = +1; int Nv = 21; 

  // Define the 3 bivariate component functions as anonymous (lamda) functions assigned to 
  // std::function objects:
  std::function<double(double, double)> fx, fy, fz;
  fx = [] (double u, double v) { return u*v; };
  fy = [] (double u, double v) { return u+v; };
  fz = [] (double u, double v) { return u-v; };
  // maybe use a more interesting surface (torus, sphere, whatever), move to Demos, maybe let the 
  // user select the surface to be drawn - update the demo that currently creates the torus data
  // itself and uses low-level functions...but maybe, we should keep the demos for how to use the 
  // low-level functions...or point the user to look at the high-level functions, if they want to
  // replicate the drawings using lower level functions....hmmm...maybe have demoTorusLowLevel
  // and demoSurface

  // plot the surface:
  GNUPlotter::plotSurface(fx, fy, fz, Nu, uMin, uMax, Nv, vMin, vMax);
}

void complexExperiment()
{
  // Set up range and umber of sampling points for real and imaginary part:
  double rMin = -3; double rMax = +3; int Nr = 31;
  double iMin = -3; double iMax = +3; int Ni = 31;

  // Define the complex function w = f(z) = z^2 as example complex function:
  function<complex<double>(complex<double>)> f;
  //f = [] (complex<double> z) { return z*z; };
  //f = [] (complex<double> z) { return z*z*z; };
  //f = [] (complex<double> z) { return z*z*z*z; };
  //f = [] (complex<double> z) { return 1./z; };         // 1st order pole at z=0

  f = [] (complex<double> z) { return z + 1./z; };
  // https://www.youtube.com/watch?v=rB83DpBJQsE
  // https://www.physik.uni-bielefeld.de/~borghini/Teaching/Hydrodynamics15/05_19.pdf
  // https://www.physik.uni-bielefeld.de/~borghini/Teaching/Hydrodynamics15/Hydrodynamics.pdf
  // -> page 61, eqIV.40: i think, we should use the negative derivative of 
  //    z + 1/z for the velocity field, not the function itself

  // https://en.wikipedia.org/wiki/Potential_flow_around_a_circular_cylinder
  // https://en.wikipedia.org/wiki/Potential_flow#Analysis_for_two-dimensional_flow

  //f = [] (complex<double> z) { return 1./(z+1.) + 1./(z-1.); };  // 2 poles at -1 and +1 (dipole..verify)
  //f = [] (complex<double> z) { return exp(z); };
  //f = [] (complex<double> z) { return sin(2.0*z); };

  // dipole: 1./(z-1.) + 1/(z+1.), maybe use 1./(z-1.) - 1/(z+1.) for source/sink (like positive 
  // and negative charge, ...but we need the inverse square law if we want to represent physical
  // situations...so our poles should be 2nd order
  // quadrupole: 1./(z-1.) + 1/(z+1.) + 1./(z-i) + 1/(z+i)

  // plot the surface corresponding to the function:
  //plotComplexSurface(f, Nr, rMin, rMax, Ni, iMin, iMax);
  //plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax);
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, true);
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, false);

  // todo: try other ways to visualize a complex function - for example by showing, how grid-lines
  // are mapped (real, imag, radial, angular)
  // how about drawing curves in the z-plane (domain) and their image curves in the w-plane (range)
}


void vectorFieldExperiment()
{
  // Create two bivariate functions that resemble the complex function 
  // w = f(z) = z^2 = (x + iy)^2
  std::function<double(double, double)> fx, fy;
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }

  // todo: draw the vector field of a dipole, move to demos, maybe let the function have a 
  // parameter that selects, which particular function is drawn (z^2, dipole, quadrupole, etc.)

  // plot the function as vector field:
  GNUPlotter::plotVectorField2D(fx, fy, 31, -3., +3., 21, -2., +2.);



}
// info for drawing vector fields:
// https://stackoverflow.com/questions/5442401/vector-field-using-gnuplot
// http://www.gnuplotting.org/vector-field-from-data-file/
// for styling the arrows, see here:
// http://www.gnuplot.info/demo/arrowstyle.html

// -maybe give the user the option to scale the arrow-lengths


/** A class to wrap various functions related to visualizing the complex function f(z) = z^2.
..maybe have similar classes for ZedCubed, ZedInverse or OneOverZed, ExpZed, SineZed, etc. */

class ZedSquared
{

public:

  // x-component of 1st solution to the first order ODE system x' = x^2 - y^2, y' = 2*x*y which can 
  // be obtained by wolfram alpha via:
  // DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == 2 x[t] y[t]}, {x, y}, t] 
  // which gives two solutions
  static double fieldX1(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double x  = -(k2 * (t-2*C2)) / (1+4*k2*C2*C2 + k2*t*t - 4*k2*t*C2);
    return x;
    // x = -(e^(2 C1) (t - 2 C2))/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2);
  }

  static double fieldY1(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double a  = 1+4*k2*C2*C2 - 4*k2*t*C2 + k2*t*t;
    double b  = t - 2*C2;
    double s  = k2-(4*k2*k2*b*b) / a*a;
    double y  = (1./2) * (k1 + sqrt(max(0.0, s)));
    return y;
    // y = 1/2 (e^(C1) + sqrt(e^(2 C1) - (4 e^(4 C1) (t - 2 C2)^2)/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
  }

  static void fieldParamLimits1(double c1, double c2, double* tMin, double* tMax)
  {
    // Computes the limits between which the value under the square root in the above function
    // s = k^2-(4*k^4*b*b) / a*a is nonnegative - this is the range, which the parameter t is allowed
    // to traverse
    // sage:
    //  var("t a b k c2")
    //  a = 1+4*k^2*c2^2 - 4*k^2*t*c2 + k^2*t*t
    //  b = t - 2*c2;
    //  s = k^2-(4*k^2*k^2*b*b) / a*a;
    //  solve(s == 0, t)
    // gives:
    //  t == 1/2*(4*c2*k - 1)/k, t == 1/2*(4*c2*k + 1)/k

    double k = exp(c1);
    *tMin = (1./2)*(4*c2*k - 1)/k;   // ...simplify/optimize
    *tMax = (1./2)*(4*c2*k + 1)/k;
  }

  // 2nd solution:
  static double fieldX2(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double x  = -(k2 * (t+2*C2)) / (1+4*k2*C2*C2 + k2*t*t + 4*k2*t*C2);
    return x;
    // x = -(e^(2 C1) (t + 2 C2))/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  }

  static double fieldY2(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double a  = 1+4*k2*C2*C2 + 4*k2*t*C2 + k2*t*t;
    double b  = t + 2*C2;
    double s  = k2-(4*k2*k2*b*b) / a*a;
    double y  = (1./2) * (k1 - sqrt(max(0.0, s)));
    return y;
    // y = 1/2 (e^(C1) - sqrt(e^(2 C1) - (4 e^(4 C1) (t + 2 C2)^2)/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
  }
  // todo: refactor to avoid code duplication...maybe try to avoid recomputations - but to fully 
  // avoid them, we may need to be able to pass a vector-valued function to the plotter instead of 
  // two scalar valued functions (the computations for x and y have terms in common)

  // maybe try to use natural parametrization (nach bogenlänge parametrisieren) - find a function to
  // apply to t to find s...or actually we need t as function of s

  static void fieldParamLimits2(double c1, double c2, double* tMin, double* tMax)
  {
    // Computes the limits between which the value under the square root in the above function
    // s = k^2-(4*k^4*b*b) / a*a is nonnegative - this is the range, which the parameter t is allowed
    // to traverse
    // sage:
    //  var("t a b k c2")
    //  a = 1 + 4*k^2*c2^2 + 4*k^2*c2*t + k^2*t^2
    //  b = 2*c2 + t
    //  s = k^2-(4*k^2*k^2*b*b) / a*a;
    //  solve(s == 0, t)
    // gives:
    //  t == -1/2*(4*c2*k + 1)/k, t == -1/2*(4*c2*k - 1)/k

    double k = exp(c1);
    *tMin = (-1./2)*(4*c2*k + 1)/k;   // ...simplify/optimize
    *tMax = (-1./2)*(4*c2*k - 1)/k;
  }

};
void addZedSquaredFieldLine(GNUPlotter& plt, double c1, bool flipY = false) 
{
  // c1 controls the size of the loop that the field line draws, flipY lets the field line flipped
  // along the x-axis (i.e. the sign of y is flipped). somehow, we don't seem to get the lower 
  // halfplane field lines from the equations, so we cheat and just add the via symmetry - figure
  // out, why we don't get them naturally from the equations - have we missed a solution to the
  // ODE system?

  static int numFieldLines = 0; // quick and dirty counter - todo: make a class, use a member

  double c2 = 0;
  // c2 seems to have no visible effect (maybe it controls the speed and therefore the range 
  // tMin..tMax?)

  double sign = 1; if(flipY) sign = -1;

  std::function<double(double)> gx, gy;
  double tMin, tMax;
  std::string s2;
  std::string s1 = "index ";
  std::string s3 = " using 1:2 with lines lt 1 notitle"; 

  int numPoints = 401; 
  // maybe this should depend on the total length of the field line - evaluate the line integral
  // for length computation

  // add data and commands for the first half of the field line:
  s2 = to_string(2*numFieldLines+1);
  ZedSquared::fieldParamLimits1(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        ZedSquared::fieldX1(t, c1, c2); };
  gy = [&] (double t) { return sign * ZedSquared::fieldY1(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3); 

  // add data and commands for the second half of the field line:
  s2 = to_string(2*numFieldLines+2);
  ZedSquared::fieldParamLimits2(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        ZedSquared::fieldX2(t, c1, c2); };
  gy = [&] (double t) { return sign * ZedSquared::fieldY2(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3);

  numFieldLines++;
}
void zedSquaredVectorField()
{
  // We plot the 2D vector field corresponding to the complex function f(z) = z^2 and also draw
  // curves that represents a field lines

  // create and set up plotter:
  GNUPlotter plt;                 // plotter object
  plt.clearCommandFile();         // we don't want to use the default line styles/colors
  plt.setGrid();
  plt.setGraphColors("209050");   // field line color, visible on top of the arrows

  // add data for the 2 bivariate functions and commands for plotting the vector field:
  std::function<double(double, double)> fx, fy; // vector field fx(x,y), fy(x,y)
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }
  //plt.addDataVectorField2D(fx, fy, 15, -1.5, 1.5, 31, -3.0, +3.0);
  plt.addDataVectorField2D(fx, fy, 21, -2.0, 2.0, 31, -3.0, +3.0);
  plt.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette notitle");
  plt.addCommand("set palette rgbformulae 30,31,32 negative");

  // add data and commands for field lines:
  double cMin = -3.0, cMax = 1.0, cStep = 0.5;
  double c = cMin;
  while(c <= cMax) {
    addZedSquaredFieldLine(plt, c, false);
    addZedSquaredFieldLine(plt, c, true);
    c += cStep;
  }
  // obtaining the field lines for the bottom halfplane by reflection is actually cheating - i 
  // think, they should somehow naturally fall out of the equations - but for some reason didn't


  plt.plot();
  // todo:
  // -add arrows and maybe something that let's use see the speed (maybe plot segments where the 
  //  particle is fast fainter - resembles an analog oscilloscope look)
  // -maybe we could add tangent/velocity vectors on the field line..this would amount to just 
  //  evaluate the vector field at these points - but if all we have is the parametric equations,
  //  we could also use numerical derivatives
}
// -how about equipotential lines? for this, we perhaps first should figure out how to draw several
//  curves on top of a scalar field in general

// todo: plot vector field of a polynomial with zeros placed at -1, +1, -i, +i, 0, maybe
// also plot vector fields of rational functions

std::complex<double> rationalVectorField(std::complex<double> z)
{
  // Computes the value of a rational function
  complex<double> i(0, 1);  // imaginary unit
  return (z-1.) * (z+1.) * (z-i) * (z+i) / z;
  // maybe try more interesting patterns of poles and zeros, try visualizing filter transfer 
  // functions as vector fields
}
void demoVectorField()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return rationalVectorField(z); };
  GNUPlotter::plotComplexVectorField(f, 31, -1.5, +1.5, 31, -1.5, +1.5, false);
}



void testInitialValueSolver()
{
  // We test the initial value solver by throwing the simple differential equation y' = k*y at it
  // and compare the produced results with the analytic result given by y(t) = exp(k*t). We also 
  // compare the different accuracies obtained by different stepper methods (Euler, midpoint, etc.)

  double k = -1.0;  // constant in y' = k*y
  double h =  0.1;  // step size
  int N = 50;       // number of datapoints

  // create data-arrays and objects:
  double s[2];      // state
  std::function<void (const double *y, double *yp)> f;
  f = [&] (const double *y, double *yp) { 
    yp[0] = 1.0;
    yp[1] = k * y[1];
  };
  std::vector<double> tA(N), tE(N), tM(N), yA(N), yE(N), yM(N);
  InitialValueSolver<double> solver;
  solver.setDerivativeFunction(f, 2);
  solver.setStepSize(h);

  // produce analytic solution for reference:
  for(int n = 0; n < N; n++) {
    tA[n] = h*n;  yA[n] = exp(k*tA[n]); }

  // produce output via Euler steps:
  s[0] = 0.0; s[1] = 1.0;
  for(int n = 0; n < N; n++) { 
    tE[n] = s[0]; yE[n] = s[1]; solver.stepEuler(&s[0], &s[0]); }

  // produce output via midpoint method:
  s[0] = 0.0; s[1] = 1.0;
  for(int n = 0; n < N; n++) { 
    tM[n] = s[0]; yM[n] = s[1]; solver.stepMidpoint(&s[0], &s[0]); }

  // compute the maximum error for each method:
  double eE, eM;  
  eE = maxError(yA, yE);  // Euler error
  eM = maxError(yA, yM);  // midpoint error
  // this works only for fixed stepsize!
  // ..ok - yes - midpoint error is much less than Euler error, as it should be
  // todo: try generalized Euler based on non-Newtonian calculus - with this, the Euler method 
  // should produce exact results


  GNUPlotter plt;
  plt.addDataArrays(N, &tA[0], &yA[0]);
  plt.addDataArrays(N, &tE[0], &yE[0]);
  plt.addDataArrays(N, &tM[0], &yM[0]);
  plt.plot();
}


void lorenzSystemDerivative(const double *y, double *yp)
{
  // parameters:
  double sigma = 10.0;
  double rho   = 28.0;
  double beta  = 8.0/3.0;

  // compute derivative vector:
  yp[0] = 1.0;                      // t' = 1 (time-axis: y[0] = t and yp[0] = y[0]' = t' = 1)
  yp[1] = sigma * (y[2] - y[1]);    // x' = sigma * (y-x)
  yp[2] = y[1]*(rho - y[3]) - y[2]; // y' = x * (rho-z) - y
  yp[3] = y[1]*y[2] - beta * y[3];  // z' = x*y - beta*z
}
void testLorenz()
{
  // Demonstrates drawing field-lines/trajectories of a Lorenz system, using the ODE solver.
  // https://en.wikipedia.org/wiki/Lorenz_system

  int N = 2000;  // number of datapoints

  typedef std::vector<double> Vec;
  Vec state(4);                   // state vector: time and position in 3D space
  Vec error(4);                   // error estimates for all variables
  Vec t(N), x(N), y(N), z(N);     // arrays for recording the ODE outputs
  Vec et(N), ex(N), ey(N), ez(N); // error estimates

  InitialValueSolver<double> solver;
  solver.setDerivativeFunction(&lorenzSystemDerivative, 4);
  solver.setAccuracy(0.2);

  // initialize state:
  state[0] = 0.0;  // time starts at zero
  state[1] = 1.0; 
  state[2] = 0.0; 
  state[3] = 0.0; 

  // iterate state and record the outputs of the ODE solver in our arrays:
  for(int n = 0; n < N; n++) {

    t[n] = state[0];  // time
    x[n] = state[1];
    y[n] = state[2];
    z[n] = state[3];

    et[n] = error[0];
    ex[n] = error[1];
    ey[n] = error[2];
    ez[n] = error[3];

    //solver.stepEuler(&state[0], &state[0]); // in-place update of the state vector
    //solver.stepMidpoint(&state[0], &state[0]);
    solver.stepMidpointAndAdaptSize(&state[0], &state[0], &error[0]);
  }

  Vec dt = diff(t); // the step-sizes taken

  // plot:
  GNUPlotter plt;                             // create plotter object
  plt.addDataArrays(N, &x[0], &y[0], &z[0]);  // pass the data to the plotter
  //plt.addCommand("set view 65,45");
  plt.addCommand("set view 80,50");
  plt.plot3D();

  GNUPlotter plt2;
  plt2.addDataArrays(N, &t[0], &x[0], &y[0], &z[0]);              // coordinates
  //plt2.addDataArrays(N, &t[0], &et[0], &ex[0], &ey[0], &ez[0]); // error estimates
  //plt2.addDataArrays(N, &t[0], &dt[0]);                         // step-sizes
  plt2.plot();
}


std::complex<double> complexDipoleField(std::complex<double> z,
  std::complex<double> cl = -1.0, std::complex<double> cr = +1.0)
{
  // Computes the field of a dipole with two charges at z = -1 and z = +1 with charge cl and cr
  // respectively (cl, cr stand for for left and right charge)
  // ...but not physically correct...

  typedef std::complex<double> Complex;

  Complex pl(-1.0, 0.0);     // position of left charge
  Complex pr(+1.0, 0.0);     // position of right charge
  double dl  = abs(z-pl);    // distance of z to left charge
  double dr  = abs(z-pr);    // distance of z to righ charge

  //double dl3 = dl*dl*dl;     // cubes of... 
  //double dr3 = dr*dr*dr;     // ...the distances
  //double dl2 = dl*dl;     // squares of... 
  //double dr2 = dr*dr;     // ...the distances
  // the physically correct law (using the cubes) doesn't look good (we need the cube because the
  // distance appears also in the numerator (hidden in the non-normalized vector))
  
  return cl*(z-pl)/dl + cr*(z-pr)/dr; // or does it have to be pl-z ?
}
void demoComplexDipole()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return complexDipoleField(z); };
  GNUPlotter::plotComplexVectorField(f, 41, -2., +2.0, 21, -1.0, +1.0, false);
}
// maybe start with the equation for the potential, the field-lines should then go into the 
// direction of the gradient - maybe this can be done automatically by computing numeric gradients
// -> have a function addPotentialFieldLines2D




class Charge2D  // not yet finished
{

public:

  Charge2D(double charge, double x, double y) : c(charge), cx(x), cy(y) {}

  /** Returns the potential at point (x,y) caused by this charge. */
  double potentialAt(double x, double y)
  {
    return c / getDistanceTo(x, y); // see formulas 4.23 and 6.8 in the feynman lectures
  }

  /** Returns x-component of electric field at point (x,y) caused by this charge. */
  double xFieldAt(double x, double y)
  {
    double r = getDistanceTo(x, y);
    return c*(x - cx) / (r*r*r);
  }
  // should equal the x-component of negative gradient of the potential
  // what if r == 0...the potential becomes infinite - but what about the field? there's no 
  // meaningful direction, it could have


  /** Returns y-component of electric field at point (x,y) caused by this charge. */
  double yFieldAt(double x, double y)
  {
    double r = getDistanceTo(x, y);
    return c*(y - cy) / (r*r*r);
  }

  /** Returns the distance between the charge's position and the given point. */
  double getDistanceTo(double x, double y)
  {
    double dx = x - cx;  // or should it be the other way around? do we want to point from x,y to
    double dy = y - cy;  // the charge or vice versa?
    double d  = sqrt(dx*dx + dy*dy);
    //return d;
    return max(d, 0.01);  // avoid singularities
  }

protected:

  double c  = 1;   // value/amount/strength of the charge
  double cx = 0;   // x-coordinate of the charge
  double cy = 0;   // y-coordinate of the charge

};

/*
Using sage to find the gradient of the potential gives this:

var("x y cx cy c")
P(x,y)  = c / sqrt((x-cx)^2 + (y-cy)^2)
Ex(x,y) = diff(P(x,y), x) 
Ey(x,y) = diff(P(x,y), y) 
Ex, Ey

Ex = c*(cx - x) / ((cx - x)^2 + (cy - y)^2)^(3/2)
Ey = c*(cy - y) / ((cx - x)^2 + (cy - y)^2)^(3/2)
*/

void addFieldLine(GNUPlotter& plt, InitialValueSolver<double>& slv, double x0, double y0, int N,
  int graphIndex)
{
  typedef std::vector<double> Vec;
  Vec s(3);                         // state vector: time and position in 2D space
  Vec t(N), x(N), y(N);             // arrays for recording the ODE outputs
  s[0] = 0; s[1] = x0; s[2] = y0;   // initial conditions
  for(int i = 0; i < N; i++) {
    t[i] = s[0];                    // not used for plot - maybe get rid..
    x[i] = s[1];
    y[i] = s[2];
    slv.stepMidpointAndAdaptSize(&s[0], &s[0]);
  }
  plt.addDataArrays(N, &x[0], &y[0]);
  plt.addGraph("index " + to_string(graphIndex) + " using 1:2 with lines lt 1 notitle"); 
}

void addRadialEquipotential(GNUPlotter& plt, 
  const std::function<double (double r, double a)>& Pra, 
  double pot, double x0, double y0, 
  int numAngles, int graphIndex)
{
  double angle;
  std::function<double (double r)> Pr;
  Pr = [&] (double r) { return Pra(r, angle); };


  std::vector<double> x(numAngles), y(numAngles);
  for(int i = 0; i < numAngles; i++)
  {
    angle = i * 2 * M_PI / (numAngles-1);
    double radius = findRoot(Pr, 0.0, 10.0, pot); // the 10 here should be a use parameter, maybe the 0 too
    x[i] = radius * cos(angle) + x0;
    y[i] = radius * sin(angle) + y0;
  }
  plt.addDataArrays(numAngles, &x[0], &y[0]);
  plt.addGraph("index " + to_string(graphIndex) + " using 1:2 with lines lt 2 notitle"); 
}


void testDipole()
{
  // Place the two charges:
  Charge2D c1(-1, -1, 0);   // negative unit charge at (x,y) = (-1,0)
  Charge2D c2(+1, +1, 0);   // positive unit charge at (x,y) = (+1,0)

  // Functions for potential and x,y components of electric field:
  std::function<double(double, double)> P, Ex, Ey; 
  P  = [&] (double x, double y) { return c1.potentialAt(x,y) + c2.potentialAt(x,y); };
  Ex = [&] (double x, double y) { return c1.xFieldAt(   x,y) + c2.xFieldAt(   x,y); };
  Ey = [&] (double x, double y) { return c1.yFieldAt(   x,y) + c2.yFieldAt(   x,y); };
  // todo: instead of defining Ex, Ey explicitly/analytically, (optionally) use a numeric gradient
  // of the potential - have a function numericPartialDerivative(func(x, y), x, y, eps)...is it
  // possible to find a formula for the numeric derivative that avoids the precision loss due to
  // subtracting two very similar numbers? 



  GNUPlotter plt;
  //plt.setGraphColors("209050");                                // field line color
  //plt.setGraphColors("000080", "800000");
  plt.setGraphColors("000000", "b0b0b0");

  //plt.addCommand("set palette rgbformulae 30,31,32 negative"); // arrow color-map - not good
  //plt.addVectorField2D(Ex, Ey, 31, -3., +3., 31, -3., +3.);  // vector field arrows
  plt.setRange(-3, 3, -3, 3);


  // Define the field function (derivative of the field lines) for the ODE solver:
  std::function<void (const double *y, double *yp)> Exy;
  Exy = [&] (const double *y, double *yp) { 
    yp[0] = 1.0;
    yp[1] = Ex(y[1], y[2]); 
    yp[2] = Ey(y[1], y[2]); 
  };

  // Set up the ODE solver:
  int N = 120;  // we should probably use a stopping criterion when the line hits the charge..
  InitialValueSolver<double> solver;
  solver.setDerivativeFunction(Exy, 3);
  solver.setAccuracy(0.002);

  // add the field lines to the plot:
  int graphIndex = 0;
  int numAngles = 40;
  double radius = 0.05;
  for(int i = 0; i < numAngles; i++)
  {
    double angle = i * 2 * M_PI / numAngles;
    double x0 = radius * cos(angle);
    double y0 = radius * sin(angle);
    solver.setStepSize( 0.01);
    addFieldLine(plt, solver, x0 + 1, y0, N, graphIndex);
    graphIndex++;
    solver.setStepSize(-0.01);
    addFieldLine(plt, solver, x0 - 1, y0, N, graphIndex);
    graphIndex++;
  }

  // draw equipotentials:
  double angle = 0.0;
  double pot = 1./1; // the potential for the equipotential line that we want to draw

  // Define potential as function of radius - then angle is taken from our local variable here:
  std::function<double (double r, double angle)> Pra1, Pra2;
  Pra1 = [&] (double r, double angle) { 
    double x = r * cos(angle) + 1;  // +1 because we measure from the right charge
    double y = r * sin(angle);
    return P(x, y);
  };
  Pra2 = [&] (double r, double angle) 
  { 
    double x = r * cos(angle) - 1; 
    double y = r * sin(angle);   
    return P(x, y);
  };


  int maxPot = 20;
  for(int i = 1; i <= maxPot; i++) {
    addRadialEquipotential(plt, Pra1,  1./i,  1.0, 0.0, 100, graphIndex); 
    graphIndex++;
    addRadialEquipotential(plt, Pra2, -1./i, -1.0, 0.0, 100, graphIndex); 
    graphIndex++;
  }

  // todo: draw equipotentials first
  





  /*
  numAngles = 100;

  // factor out:
  std::vector<double> x(numAngles), y(numAngles);
  for(int i = 0; i < numAngles; i++)
  {
    angle  = i * 2 * M_PI / (numAngles-1);
    radius = findRoot(Pr, 0.0, 10.0, pot);
    x[i] = radius * cos(angle) + 1;
    y[i] = radius * sin(angle);
  }
  plt.addDataArrays(numAngles, &x[0], &y[0]);
  plt.addGraph("index " + to_string(graphIndex) + " using 1:2 with lines lt 1 notitle"); 
  graphIndex++;
  */
  



  // draw circles for the charges:
  plt.addCommand("set object 1 circle at  1,0 size 0.12 fc rgb \"white\" fs solid 1.0 front"); 
  plt.addCommand("set object 2 circle at  1,0 size 0.12 fc rgb \"black\" front"); 
  plt.addCommand("set object 3 circle at -1,0 size 0.12 fc rgb \"white\" fs solid 1.0 front"); 
  plt.addCommand("set object 4 circle at -1,0 size 0.12 fc rgb \"black\" front");

  // draw plus and minus onto the charges:
  plt.addCommand("set arrow 1 from -0.95,0 to -1.05,0 nohead lw 2 fc rgb \"black\" front");
  plt.addCommand("set arrow 2 from 0.95,0 to 1.05,0 nohead lw 2 fc rgb \"black\" front");
  plt.addCommand("set arrow 3 from 1,-0.05 to 1,0.05 nohead lw 2 fc rgb \"black\" front");
  // ...the plus looks a bit asymmetric - why?


  plt.setPixelSize(600, 600);
  plt.addCommand("set size square"); 
  plt.plot();

  // todo: 
  // -add equipotentials (requires implicit equation solver and some additional stuff)
  //  -maybe keep one coordinate fixed (but how to choose it?) and use 1D bisection
  //  -perhaps we should use angle and radius (from the charge) as the two coordinates - then we
  //   would again choose equidistant angles
}

// try to create a plot like the one at the bottom here:
// http://www.feynmanlectures.caltech.edu/II_04.html
// with field-lines and equipotential lines - ideally, we should just pass the function a potential
// function, i.e. a scalar function of two variables
// P = []  (double x, double y) { return chargePotential(-1, -1, 0) + chargePotential(+1, +1, 0); };
// 1st input: charge, 2nd: x-coord, 3rd: y-coord
// Ex = dP/dx, Ey = dP/dy (numeric derivatives)
// plotPotentialField
// subfunctions: addFieldLines, addEquipotentials
// maybe for the equipotentials, we need a way to plot a curve defined by an implicit equation
// f(x,y) = c




/*

Ideas:
-plot a bunch of field-lines - maybe use as example 2 charges at (-1,0) and (+1,0), start the 
 field-lines in the middle vertical line (0, y) and from there, integrate them numerically into
 both directions...maybe using a function addBiDirectionalFieldLine
-make a demo showing a positive charge at +1 and a negative charge at -1 using the physically 
 correct law for the electric field - make another plot with two negative and two positive charges
 the two negative charges should look like 2 gravitational fields - call it demoDipole

-plot complex mapping - maybe there are various ways to do this...


-plot 3D curves (trefoil knot)
-plot 3D vector fields
-plot curves on top of a 2D vector field - can be used to show equipotential curves and/or 
 integration paths

 weitz on visualization:
 https://www.youtube.com/watch?v=BhtnlKOC-0s&t=189s
 https://www.youtube.com/watch?v=BhtnlKOC-0s


Field lines:
-draw field lines of 2D vector fields - how do we find them? start somewhere and follow the field?
 ...but what about error accumulation? ...in any case, it seems we need a general way to add many
 curves to a vector field (for field lines of vector fields and equipotential lines for scalar
 fields)
 https://www.quora.com/How-can-I-find-an-expression-for-vector-field-lines
 https://math.stackexchange.com/questions/1992208/how-to-find-the-field-lines-of-a-vector-field
-maybe we should plot the field lines as parametric curves whose equations should be obtained
 analytically? maybe like this:
 -let f(x,y) and g(x,y) be the functions that define our vector field
 -we want a family of parametric curves x(t), y(t) such that in any point x,y on one of the curves, 
  the curve's tangent is in the direction of the vector field - so we require: 
  x'(t) = f(x(t),y(t)), y'(t) = g(x(t),y(t)) for any t, or shorter: x' = f(x,y), y' = g(x,y)
  -> this is a system of 2 first order differential equations
  -> maybe if f and g are simple enough, we can find an analytic solution to this system?
  -> if, not - use a numerical initial value solver:
     -choose a couple of strategically placed points in the plane (for example, equidistant on
      one of the axes) and just follow the field for positive and negative times
     -if start and end-points are known (such as with a dipole), maybe a boundary value solver
      can be used? 
      -for the dipole, we know that lines start at the source and end at the sink
      -we may also have to select an initial direction
     -maybe the function object passed to the plotter that generates the field lines is the solver?
      or maybe i should pre-solve the equation and pass some sort of InterpolatingFunction object
      (makes more sense, since the plotting routines want random access functions)
-for the complex mapping w = f(z) = z^2 interpreted as vector field, this leads to the first order
 ODE system x' = x^2 - y^2, y' = 2*x*y where the prime denotes differentiation with respect to the 
 parameter t.
 wolfram alpha can solve this with the command:
 DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == 2 x[t] y[t]}, {x, y}, t]
 which reults in 2 solutions:

  x = -(e^(2 C1) (t - 2 C2))/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  y = 1/2 (e^(C1) + sqrt(e^(2 C1) - (4 e^(4 C1) (t - 2 C2)^2)/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))

  x = -(e^(2 C1) (t + 2 C2))/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  y = 1/2 (e^(C1) - sqrt(e^(2 C1) - (4 e^(4 C1) (t + 2 C2)^2)/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))

  ...try plotting them for various values of C1, C2 ...maybe to simplify, define 
  k1 = e^C1, k2 = e^C2, k12 = k1^2 = e^(2 C1), etc.

  do the same for x' = x^2 - y^2, y' = -2*x*y - the additional minus sign accounts for complex 
  conjugation of the Polya vector field associated to f(z) = z^2:
  DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == -2 x[t] y[t]}, {x, y}, t]
  ...but wolfram alpha doesn't understand this - wtf - the only difference is a minus sign!
  ok - wolfram cloud can do it - and the result is *much* more complicated than without the minus
  sign - it's still strange that alpha says that it doesn't understand the question - it should
  say something like: the problem is too complicated for alpha

  can we find a scalar valued function of x,y whose gradient is our desired vector field, i.e. a 
  potential function for our vector field?






  -> integrate with respect to t


  so it seems we
  must partially integrate f and g to obtain: 
  x(t) = integral_a^t f(x,y) dx, y(t) = integral_b^t g(x,y) dy
  for some constants a,b for the lower limits...is that correct? it seems reasonable to have two
  free parameters a,b since for each point on the plane, there should be exactly one field line
  that passes through it - so if we pick apoint x,y in the plane, we should be able to adjust a,b
  to find a field line through it (however that same field line passes through many points..hmm)




-how would we represent a scalar field in 3D? maybe as (semi-transparent) spheres with a size 
 representing the value? or maybe the transparency should represent the value? or both? ..or maybe
 cubes instead of spheres?

idea to visualize a rank-2 tensor field in 2D (i.e. a 2x2 matrix-field defined in the xy-plane)
-at each sample point, draw a small 2x2 square and color the 4 quadrants of that square according 
 to a colormap
-the size of squares may be such that when the sampling distances are dx, dy, a square centered at 
 x,y may extend from x-dx/2 to x+dx/2 in the x-direction and from y-dy/2 to y+dy/2 in the 
 y-direction - but we may artificially shrink these squares to have some margin between them
-a constant tensor field given by the 2x2 identity matrix will result in diagonal stripes of 
 high-value from top-left to bottom-right
-maybe it can be done via the matrix dataset and image graphing functions (similar to heat-maps 
 or spectrograms)
-maybe several such patterns with different sampling intervals can be overlaid to convey coarse
 and fine structure...maybe the coarse ones should be smoothed - or better: evaluated at an 
 oversampled grid and accumulated
 
-maybe try it with the Jacobian of f(z) = z^2 - the pattern should show symmetries due to the 
 Cauchy-Riemann conditions, try also the metric tensor of polar coordinates


-todo: draw a trefoil-knot, a mobius-strip and a trefoil-mobius-knot
 -for a pramatrization of the mobius-strip, see here: https://www.youtube.com/watch?v=dz7y7mFLW3U
 -for the mobius knot, replace the circle with the trefoil knot


*/