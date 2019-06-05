#include "Demos.h"
//#include <complex> // may be removed, if it will be included in GNUPlotter.h someday
#define M_PI 3.14159265358979323846

using namespace std;

// define some functions for plotting (move to a file ExampleFunctions.h/cpp):

double identity(double x)
{
  return x;
}

double square(double x)
{
  return x*x;
}

double sincUnscaled(double x)
{
  if(fabs(x) < DBL_EPSILON)
    return 1.0;
  else
    return sin(x) / x;
}

double sinc(double x)
{
  return sincUnscaled(M_PI*x);
}

double sinc2D(double x, double y)
{
  if(fabs(x*y) < DBL_EPSILON)
    return 1.0;
  else
    return sin(M_PI*x)*sin(M_PI*y) / (M_PI*M_PI*x*y);
}

double sincRadial(double x, double y)
{
  return sinc(sqrt(x*x+y*y));
}

double gauss2D(double x, double y)
{
  return exp(-(x*x+y*y)); 
}

//-------------------------------------------------------------------------------------------------

void demoArrayPlot()
{
  // Demonstrates, how to plot arrays of values against their index. y1 is the Fibonacci 
  // sequence, y2 a similar sequence that uses different start values for the recursion.

  // generate our arrays to plot:
  static const int N = 10;
  int y1[N], y2[N];
  y1[0] = 1;
  y1[1] = 1;
  y2[0] = 0;
  y2[1] = 1;
  for(int n = 2; n < N; n++)
  {
    y1[n] = y1[n-1] + y1[n-2];
    y2[n] = y2[n-1] + y2[n-2];
  }

  // plot them:
  GNUPlotter p;
  p.plotArrays(N, y1, y2);  // use static function later
}

void demoFunctionTablePlot()
{
  // Demonstrates plotting of 3 functions (given as data-arrays) against a common x-axis.

  // create x-axis and 3 functions:
  static const int N = 300;                 // number of data points
  double x[N], y1[N], y2[N], y3[N];
  GNUPlotter::rangeLinear(x, N, 0.0, 1.5);  // create x-axis values 
  for(int n = 0; n < N; n++)
  {
    y1[n] = x[n];                           // y1 = x
    y2[n] = x[n]*x[n];                      // y2 = x^2
    y3[n] = sqrt(x[n]);                     // y3 = sqrt(x)
  }

  // plot:
  GNUPlotter p;
  p.plotFunctionTables(N, x, y1, y2, y3); 
}

void demoFunctionPlot()
{
  GNUPlotter p;
  p.plotFunctions(1000, -5.0, 10.0, &sin, &cos);
}

void demoAliasing()
{
  // Create 2 pseudocontinuous (i.e. densely sampled) sine waves with frequencies 3 Hz and 7 Hz and
  // then sample the 3 Hz sine with a samplerate of 10 Hz. We show in the plot that the samples of 
  // the 3 Hz sine would indeed be the same values if we had sampled a 7 Hz sine (with opposite 
  // phase) - that shows, that, at 10 Hz samplerate, a 7 Hz sine would alias into a 3 Hz sine:

  static const int N1 = 1001;                  // number of datapoints for continuous signals
  static const int N2 = 11;                    // number of samples
  double tMin = 0;                             // start time
  double tMax = 1;                             // end time
  double fs = 10;                              // sample rate
  double f1 = 3;                               // frequency of correctly sampled sinusoid
  double f2 = 7;                               // frequency of aliased sinusoid
  double t[N1];                                // dense (pseudo)continuous time axis
  double ts[N2];                               // sampled time axis
  double x1[N1], x2[N1];                       // continuous sinusoids
  double xs[N2];                               // sample values
  GNUPlotter::rangeLinear(t,  N1, tMin, tMax); // fill (pseudo)continuous time axis
  GNUPlotter::rangeLinear(ts, N2, tMin, tMax); // fill sampled time axis
  double w1 = 2*M_PI*f1;                       // normalized radian frequency of 1st sinusoid
  double w2 = 2*M_PI*f2;                       // ditto for 2nd sinusoid
  int n;
  for(n = 0; n < N1; n++)
  {
    x1[n] =  sin(w1*t[n]);                     // value of 1st sinusoid
    x2[n] = -sin(w2*t[n]);                     // value of 2nd sinusoid
  }
  for(n = 0; n < N2; n++)
    xs[n] = sin(w1*ts[n]);                     // sampled sinusoid value

  // create and set up plotter object: 
  GNUPlotter p;
  p.addDataArrays(N1, t,  x1, x2);               // add two continuous sines as 1st dataset
  p.addDataArrays(N2, ts, xs);                   // add sampled sine as 2nd dataset
  p.setGrid(); 
  p.setPixelSize(800, 250); 
  p.addCommand("set key opaque box");            // graphs shouldn't obscure legends
  p.addCommand("set xtics 0.1");                 // x-gridlines at 0.1, 0.2, ...
  p.setAxisLabels("Time [seconds]", "Voltage");
  p.setDashType(2, "(3,3)");                     // 7 Hz sine is dashed
    
  // The 1st graph uses column 1 of dataset 1 (index 0) for the x-axis and column 2 for the y-axis. 
  // The 2nd graph uses column 1 of the same dataset for the x-axis as well but column 3 for the 
  // y-axis. The 3rd and 4th graph use both column 1 and 2 of dataset 2 (index 1) but display it 
  // once with points and once with impulses, so we get these impulses-with-points that are 
  // commonplace in the DSP literature for representing discrete time signals:
  p.addGraph("index 0 using 1:2 with lines lw 2 lc rgb \"#0000D0\" title \"f=3\"");
  p.addGraph("index 0 using 1:3 with lines lw 1.5 lc rgb \"#A00000\" title \"f=7\"");
  p.addGraph("index 1 using 1:2 with points pt 7 ps 1.25 lc rgb \"#000000\" title \"samples\"");
  p.addGraph("index 1 using 1:2 with impulses lw 3 lc rgb \"#000000\" notitle");
  p.plot();
}

// move to helper functions:
// Computes poles of a Butterworth prototype filter of order N
vector<complex<double>> butterworthPoles(int N)
{
  vector<complex<double>> p(N);
  for(int n = 0; n < N; n++)
  {
    double k   = n+1;
    double arg = M_PI*(2*k-1)/(2*N);
    p[n] = complex<double>(-sin(arg), cos(arg));
  }
  return p;
}

// Unwraps values in the length-N array "a" with respect to a periodicity of "p".
void unwrap(double *a, int N, double p)
{
  int k = 0;
  for(int n = 1; n < N; n++)
  {
    while(fabs((a[n]+(k*p))-a[n-1]) > fabs((a[n]+((k+1)*p))-a[n-1]))
      k++;
    while(fabs((a[n]+(k*p))-a[n-1]) > fabs((a[n]+((k-1)*p))-a[n-1]))
      k--;
    a[n] += k*p;
  }
}

// Evalutes polynomial defined by its roots r at the value z
complex<double> polynomialByRoots(complex<double> z, vector<complex<double>> r)
{
  complex<double> w = 1;
  for(int i = 0; i < r.size(); i++)
    w *= z-r[i];
  return w;
}

// Evaluates complex transfer function defined by its zeros z, poles p and gain k at the complex 
// value s 
complex<double> transferFunctionZPK(complex<double> s, vector<complex<double>> z, 
  vector<complex<double>> p, double k)
{
  complex<double> num = polynomialByRoots(s, z);
  complex<double> den = polynomialByRoots(s, p);
  return k * num/den;
}

void demoFrequencyResponse()
{
  // Shows how to make use of two y-axes (for magnitude and phase) and how to create log-scaled 
  // axes. As an example, we plot magnitude and phase responses for Butterworth filters of various
  // orders 1 to 5.

  static const int N = 501;        // number of datapoints
  static const int M = 5;          // maximum order
  double wMin = 0.0625;            // minimum radian frequency
  double wMax = 16.0;              // maximum radian frequency
  double w[N];                     // radian frequency axis
  double mag[M][N], phs[M][N];     // magnitudes and phases
  double *pMag[M], *pPhs[M];       // pointers to 1D arrays

  // fill frequency axis with equally spaced values on logarithmic scale:
  GNUPlotter::rangeLogarithmic(w, N, wMin, wMax);

  // assign pointer arrays:
  int n, m;
  for(m = 0; m < M; m++)
  {
    pMag[m] = mag[m];
    pPhs[m] = phs[m];
  }

  // compute frequency response data:
  complex<double> s;               // value on s-plane where we evaluate H
  complex<double> H;               // complex frequency response H(s)
  vector<complex<double>> poles;   // Butterworth filter poles
  vector<complex<double>> zeros;   // filter zeros (empty vector)
  complex<double> j(0.0, 1.0);     // imaginary unit
  for(m = 0; m < M; m++)
  {
    poles = butterworthPoles(m+1);
    for(n = 0; n < N; n++)
    {
      H = transferFunctionZPK(j*w[n], zeros, poles, 1); // evaluate H(s) at s=j*w
      mag[m][n] = 20*log10(abs(H));                     // magnitude in dB
      phs[m][n] = 180*arg(H)/M_PI;                      // phase in degrees
      unwrap(phs[m], N, 360);                           // unwrap phase response
    }
  }

  GNUPlotter p;
  p.addDataArrays(N, w, M, pMag);
  p.addDataArrays(N, w, M, pPhs);
  p.setPixelSize(600, 350); 
  p.setTitle("Butterworth Frequency Responses for Orders 1-5");
  p.setGraphColors("A00000", "909000", "008000", "0000A0", "800080",
                   "A00000", "909000", "008000", "0000A0", "800080" );
  p.addCommand("set logscale x");
  p.addCommand("set xrange  [0.0625:16]");
  p.addCommand("set yrange  [-100:0]");
  p.addCommand("set y2range [-450:0]");
  p.addCommand("set xlabel \"Frequency in kHz\"");
  p.addCommand("set ylabel \"Magnitude in dB\"");
  p.addCommand("set y2label \"Phase in Degrees\"");
  p.addCommand("set xtics 2");    // factor 2 between (major) frequency axis tics
  p.addCommand("unset mxtics");   // no minor tics for frequency axis
  p.addCommand("set ytics 10");   // 10 dB steps for magnitude axis
  p.addCommand("set y2tics 45");  // 45° steps for phase axis

  // add magnitude graphs:
  p.addGraph("i 0 u 1:2 w lines lw 1.5 axes x1y1 notitle");
  p.addGraph("i 0 u 1:3 w lines lw 1.5 axes x1y1 notitle");
  p.addGraph("i 0 u 1:4 w lines lw 1.5 axes x1y1 notitle");
  p.addGraph("i 0 u 1:5 w lines lw 1.5 axes x1y1 notitle");
  p.addGraph("i 0 u 1:6 w lines lw 1.5 axes x1y1 notitle");

  // add phase graphs:
  p.addGraph("i 1 u 1:2 w lines lw 1.5 axes x1y2 notitle");
  p.addGraph("i 1 u 1:3 w lines lw 1.5 axes x1y2 notitle");
  p.addGraph("i 1 u 1:4 w lines lw 1.5 axes x1y2 notitle");
  p.addGraph("i 1 u 1:5 w lines lw 1.5 axes x1y2 notitle");
  p.addGraph("i 1 u 1:6 w lines lw 1.5 axes x1y2 notitle");

  p.plot();
}

// returns maximum absolute value of all real an imaginary parts
double maxAbsReIm(const vector<complex<double>>& x)
{
  double m = 0.0;
  for(int i = 0; i < x.size(); i++)
  {
    if(fabs(x[i].real()) > m)
      m = fabs(x[i].real());
    if(fabs(x[i].imag()) > m)
      m = fabs(x[i].imag());
  }
  return m;
}

// returns true, if the relative distance between x and y is smaller than the given threshold 
// ("relative" with respect to the actual absolute values of x and y, such that for larger values 
// the tolerance also increases)
bool almostEqual(complex<double> x, complex<double> y, double thresh)
{
  return abs(x-y) / fmax(abs(x), abs(y)) < thresh;
}

// Given an array of complex values z (for example, roots of a polynomial), this function plots
// their multiplicities at their positions
void drawMultiplicities(const vector<complex<double>>& z, double thresh, GNUPlotter *p)
{
  size_t N = z.size();               // number of values
  vector<complex<double>> zd(N);  // collected distinct values
  vector<int> m(N);               // m[i] = multiplicity of value zd[i]
  vector<bool> done(N);           // vector of flags, if z[i] was already absorbed into zd
  int i, j;
  int k = 0;

  // collect distinct values and their multiplicities:
  for(i = 0; i < N; i++)
  {
    if(!done[i])
    {
      zd[k]   = z[i];
      m[k]    = 1;
      done[i] = true;
      for(j = i+1; j < N; j++)  // find values equal to zd[k] == z[i]
      {
        if(!done[j] && almostEqual(z[i], z[j], thresh))
        {
          m[k]    += 1;
          done[j]  = true;
        }
      }
      k++;
    }
  }

  // k is now the number of distinct values stored in zd with associated multiplicities in m
  for(i = 0; i < k; i++)
  {
    if(m[i] > 1)
      p->addAnnotation(zd[i].real(), zd[i].imag(), " " + to_string(m[i]), "left");
  }
}

// plots poles and zeros of a filter transfer function. If zDomain is true, a unit circle will be
// shown (maybe we should draw a polar grid then)
// move to the main GNUPlotCPP file (as convenience function)
void poleZeroPlot(const vector<complex<double>>& poles, const vector<complex<double>>& zeros, 
  bool zDomain = false)
{
  GNUPlotter p;
  p.addDataComplex(poles);
  p.addDataComplex(zeros);


  // maybe draw unit circe in case of z-plane poles/zeros, maybe use polar grid, maybe indicate
  // pole/zero multiplicities

  if(zDomain == true)
  {
    // draw polar grid - hmm, doesn't look very good:
    //p.addCommand("set angles degrees"); 
    //p.addCommand("set grid polar 15 lt 1 lc rgb \"#A0A0A0\""); 


    // draw unit circle:
    p.addCommand("set object 1 ellipse at first 0,0 size 2,2 fs empty border rgb \"#808080\""); 
     // an ellipse with center 0,0 and width/height of 1 measured in the 1st coordinate system 
     // given by the bottom and left axes - we need an ellipse to make sure it always represents
     // the unit circle, even when scaling of x- and y-axis is different (it remains a circle in
     // the 1st coordinate system, even, if we zoom in)
  }

  // show the multiplicities of poles and zeros:
  double thresh = 1.e-8; // threshold for considering close poles/zeros as multiple root
  drawMultiplicities(poles, thresh, &p);
  drawMultiplicities(zeros, thresh, &p);


  p.addCommand("set size square");           // set aspect ratio to 1:1
  p.addCommand("set xzeroaxis lt 1");        // draw x-axis
  p.addCommand("set yzeroaxis lt 1");        // draw y-axis
  p.addCommand("set xlabel \"Real Part\"");
  p.addCommand("set ylabel \"Imaginary Part\"");
  p.setPixelSize(400, 400);

  // set range:
  double a = fmax(maxAbsReIm(poles), maxAbsReIm(zeros));
  a  = fmax(1.0, a);
  a *= 1.1;
  p.setRange(-a, a, -a, a);


  p.setGraphColors("000000", "000000");             // both datasets black

  p.addGraph("i 0 u 1:2 w points pt 2 ps 1 notitle");
  p.addGraph("i 1 u 1:2 w points pt 6 ps 1 notitle");

  //p.addGraph("i 0 u 1:2 w points pt 2 ps 1 title \"poles\"");
  //p.addGraph("i 1 u 1:2 w points pt 6 ps 1 title \"zeros\"");

  p.plot();

  int dummy = 0;
}

void demoPoleZeroPlotS()
{
  // We create a plot of poles and zeros of a Butterworth low-shelving filter. The zeros are scaled
  // versions of the poles.
  int    N  = 10;    // order of the filter
  double a  = 0.75;  // scaler for the zeros
  vector<complex<double>> p = butterworthPoles(N);
  vector<complex<double>> z = p;
  for(int i = 0; i < N; i++)
    z[i] *= a;
  poleZeroPlot(p, z, false);
}

void demoPoleZeroPlotZ()
{
  // We plot some arbitrarily placed poles and zeros in the z-plane, some of them having 
  // multiplicities
  vector<complex<double>> p(9), z(5);
  p[0] = complex<double>(0.6, 0.6);
  p[1] = conj(p[0]);
  p[2] = p[0];
  p[3] = p[1];
  p[4] = p[0];
  p[5] = p[1];
  p[6] = complex<double>(-0.7, 0.5);
  p[7] = conj(p[6]);
  p[8] = -0.3;
  z[0] = -1;
  z[1] = -1;
  z[2] = complex<double>(0.8, 0.6);
  z[3] = conj(z[2]);
  z[4] = 0.2;
  poleZeroPlot(p, z, true);
}

void demoTransferMagnitude()
{

}

void demoPlottingStyles()
{
  // Demonstrates, how to use various plotting styles, partly using more than 2 columns of data
  // per graph, to show things like errorbars.

  static const int N = 30;      // maximum number of datapoints
  double a = 0.3;               // multplier for the sine's argument
  double b = 1.0;               // y-axis offset between datasets
  double d[17][N];              // dataset - 17 columns of N values
              
  // create the data for the datasets:
  int i;
  double *pd[17];          // we need a pointer-to-pointer to pass
  for(i = 0; i < 17; i++)
    pd[i] = d[i];
  for(i = 0; i < N; i++)
  {
    double x = i;              
    double y = sin(a*x);

    d[0][i] = x;           // x for all datasets (except 1-column which has implicit x)

    d[1][i] = y;           // y for 1-column dataset

    y += b;
    d[2][i] = y;           // y for 2-column dataset

    y += b;
    d[3][i] = y;           // y for 3-column dataset (for yerrorlines)
    d[4][i] = 0.3;         // y-delta

    y += b;
    d[5][i] = y;           // y for 4-column dataset (for yerrorlines)
    d[6][i] = y - 0.4;     // y min
    d[7][i] = y + 0.2;     // y max

    y += b;
    d[8][i]  = y;          // box min for 5-column dataset (for candlesticks)
    d[9][i]  = y - 0.2;    // whisker min
    d[10][i] = y + 0.4;    // whisker max
    d[11][i] = y + 0.2;    // box max

    y += b;
    d[12][i] = y;          // y for 6-column dataset (for xyerrorlines)
    d[13][i] = x - 0.4;    // x min
    d[14][i] = x + 0.5;    // x max
    d[15][i] = y - 0.2;    // y min
    d[16][i] = y + 0.3;    // y max
  }

  // Create plotter, set it up and add the data:
  GNUPlotter p;
  p.setRange(0, 30, -2, 7);
  p.addCommand("set key opaque box"); // graphs shouldn't obscure legends
  p.addData(N, 17, pd);

  // Add graphs, tell the plotter which columns to use for each graph and plot:
  p.addGraph("index 0 using 2 with lines lc rgb \"#000000\" title \"lines\"");
  p.addGraph("index 0 using 1:3 with linespoints lc rgb \"#000000\" title \"linespoints\"");
  p.addGraph("index 0 using 1:4:5 with yerrorlines lc rgb \"#000000\" title \"yerrorlines\"");
  p.addGraph("index 0 using 1:6:7:8 with yerrorlines lc rgb \"#000000\" title \"yerrorlines\"");
  p.addGraph("index 0 using 1:9:10:11:12 with candlesticks lc rgb \"#000000\" title \"candlesticks\"");
  p.addGraph("index 0 using 1:13:14:15:16:17 with xyerrorlines lc rgb \"#000000\" title \"xyerrorlines\"");

  // invoke GNUPlot:
  p.plot();
}

void demoTrigFunctions()
{
  // Demostrates the general setup of the plotter object, like setting the window's pixel size, 
  // plotting range, axis labels, etc. As an example, we plot the 3 functions sin, cos, tan using
  // the facilities to create the datafile from c-functions.

  int    N    = 201;                                          // number of datapoints
  double xMin = 0;                                            // x-axis minimum value
  double xMax = 10;                                           // x-axis maximum value
  GNUPlotter p;                                               // create a plotter object
  p.setTitle("Trigonometric Functions");                      // caption for the plot
  p.setLegends("y=sin(x)", "y=cos(x)", "y=tan(x)");           // legends for the 3 graphs
  p.setAxisLabels("x-axis", "y-axis");                        // labels for the axes
  p.setPixelSize(800, 400);                                   // pixel size for plot
  p.setRange(xMin, xMax, -2, 2);                              // range for x- and y-axis
  p.setGraphColors("800000", "008000", "000080");             // red, green and blue
  p.setDashType(3, "(1,8,5,8)");                              // use dash-pattern for tan
  p.setGraphStyles("lines lw 2", "lines lw 2", "lines lw 1"); // linewidths are 2,2,1
  p.plotFunctions(N, xMin, xMax, &sin, &cos, &tan);           // plot the functions
}

void demoSquare()
{
  GNUPlotter p;
  p.addDataFunctions(501, 0.0, 5.0, &square);
  p.addDataFunctions(6,   0.0, 5.0, &square);
  p.addGraph("index 0 using 1:2 with lines lw 2 lc rgb \"#808080\" notitle");
  p.addGraph("index 1 using 1:2 with points pt 7 ps 1.2 lc rgb \"#000000\" notitle");
  p.plot();
}

void demoMatrixData()
{
  // Demonstrates, how to add matrix formatted datasets to the datafile and plot the datasets. As
  // example, we use the two functions that compute the total number of values that have to be 
  // written into a dataset as function of N and M (the lengths of the x- and y-array minus 1).
  // So, this demo shows how to use the matrix data format and its output plot shows you why - you
  // can see the data reduction factor as function of N and M. The ratio r between the size of the 
  // triplet dataset and the size of the matrix dataset (measured in number of values stored) is 
  // given by:
  // r(N,M) = 3*N*M / (N*M+N+M+1).
  // The reduction is greatest when N=M. For that case, the data reduction factor as function of 
  // N is given by: 
  // r(N) = 3*N^2 / (N^2+2*N+1) 
  // which approaches 3 for large N

  static const int Nx = 10;       // length of x-array
  static const int Ny = 10;       // length of y-array
  static const int N  = Nx-1;
  static const int M  = Ny-1;

  // memory allocation and grid initialization:
  int i, j;
  GNUPlotter p;
  double x[Nx], y[Ny], zt[Nx][Ny], zm[Nx][Ny], zr[Nx][Ny];
  double *pzt[Nx], *pzm[Nx], *pzr[Nx];
  for(i = 0; i < Nx; i++)
  {
    pzt[i] = zt[i];
    pzm[i] = zm[i];
    pzr[i] = zr[i];
  }
  GNUPlotter::rangeLinear(x, Nx, 1.0, double(Nx));
  GNUPlotter::rangeLinear(y, Ny, 1.0, double(Ny));

  // create z data matrices:
  for(i = 0; i < Nx; i++)
  {
    for(j = 0; j < Ny; j++)
    {
      zt[i][j] = 3*x[i]*y[j];                 // number of values for triplet storage
      zm[i][j] = x[i]*y[j] + x[i] + y[j] + 1; // number of values for matrix storage
      zr[i][j] = zt[i][j] / zm[i][j];         // data reduction factor
    }
  }

  // add the data (matrix datasets must come first):
  p.addDataMatrix( Nx, Ny, x, y, pzm);
  p.addDataMatrix( Nx, Ny, x, y, pzr);
  p.addDataGrid(Nx, Ny, x, y, pzt);

  // define the graphs:
  p.addGraph("i 2 w lines title \"Triplets\"");
  p.addGraph("i 0 nonuniform matrix w lines title \"Matrix\"");
  p.addGraph("i 1 nonuniform matrix w lines title \"Ratio\"");

  // set up and plot:
  p.addCommand("set hidden3d");
  p.addCommand("set view 75,55");
  p.setGraphColors("000000", "000000", "000080", "000080", "006000", "006000");
  p.setAxisLabels("N", "M", "S");
  p.plot3D();
}

void demoLissajous()
{
  // Demonstrates, how to draw a parametric curve in 2D space, using a Lissajous figure as example.

  // user parameters:
  double a = 3.0;             // number of cycles in x(t)
  double b = 2.0;             // number of cycles in y(t)
  static const int N = 1001;  // number of xy-pairs

  // Create data:
  double t;
  double x[N], y[N];
  for(int n = 0; n < N; n++)
  {
    t    = 2*M_PI*n / (N-1);
    x[n] = sin(a*t);
    y[n] = sin(b*t);
  }

  // Plot:
  GNUPlotter p; 
  p.setPixelSize(500, 500);
  p.addCommand("set size 1,1");  // to have an aspect ratio of 1:1
  p.addDataArrays(N, x, y);
  p.plot();
}

void demoLissajous3D()
{
  // Demonstrates, how to draw a parametric curve in 3D space, using a 3D Lissajous figure as 
  // example.

  // user parameters:
  double a = 2.0;             // number of cycles in x(t)
  double b = 3.0;             // number of cycles in y(t)
  double c = 5.0;             // number of cycles in z(t)
  static const int N = 1001;  // number of datapoints
                            
  // Create data:
  double t;
  double x[N], y[N], z[N];
  for(int n = 0; n < N; n++)
  {
    t    = 2*M_PI*n / (N-1);
    x[n] = sin(a*t);
    y[n] = sin(b*t);
    z[n] = sin(c*t);
  }

  // Plot:
  GNUPlotter p; 
  p.setPixelSize(500, 500);
  //p.addCommand("set size 1,1");  // to have an aspect ratio of 1:1
  p.addDataArrays(N, x, y, z);
  p.plot3D();
}



void demoHelix()
{
  // Demonstrates how to draw a parametric curve in 3D space, using a helix as example.

  // Generate the values for the parameter t:
  static const int N = 1001;                  // number of datapoints
  double tMin = 0;                            // minimum value for parameter t
  double tMax = 50;                           // maximum value for parameter t
  double t[N], x[N], y[N], z[N];              // arrays for t,x,y,z
  GNUPlotter::rangeLinear(t, N, tMin, tMax);  // fill t-array with equidistant values

  // Generate the helical curve:
  for(int i = 0; i < N; i++)
  {
    x[i] = cos(t[i]);                         // x(t) = cos(t)
    y[i] = sin(t[i]);                         // y(t) = sin(t)
    z[i] = t[i];                              // z(t) = t
  }

  // plot:
  GNUPlotter p;                               // create plotter object
  p.addDataArrays(N, x, y, z);                // pass the data to the plotter
  p.addCommand("set view 60,320");            // set up perspective
  p.plot3D();                                 // invoke GNUPlot
}

void demoPhasor()
{
  // Demonstrates how to draw a parametric curve in 3D space, using a decaying complex phasor as 
  // example.

  // Generate the values for the parameter t:
  static const int N = 1001;
  double tMin = 0;
  double tMax = 100;
  double d    = 0.03;  // decay rate
  double t[N], x[N], y[N], z[N];
  GNUPlotter::rangeLinear(t, N, tMin, tMax);

  // Generate the curve:
  double r; // radius
  for(int i = 0; i < N; i++)
  {
    r    = exp(-d*t[i]);
    x[i] = r*cos(t[i]);
    y[i] = r*sin(t[i]);
    z[i] = t[i];
  }

  // plot:
  GNUPlotter p;
  p.addDataArrays(N, x, z, y);
  p.addCommand("set view 90,85"); 
  p.plot3D();

}

void demoTorus()
{
  // Demonstrates, how to draw a 2-parametric surface in a 3-dimensional space, using a torus as 
  // example.

  // user parameters:
  double R = 2.0;                                // major radius
  double r = 0.5;                                // minor radius
  static const int Nu = 41;                      // number of grid lines around the major radius
  static const int Nv = 21;                      // number of grid lines around the minor radius

  // create parameter arrays u and v:
  double u[Nu], v[Nv];                           // arrays for parameters u and v
  GNUPlotter::rangeLinear(u, Nu, 0.0, 2*M_PI);   // fill u-array with equidistant values
  GNUPlotter::rangeLinear(v, Nv, 0.0, 2*M_PI);   // fill v-array with equidistant values

  // Create the data vector. The outer index runs over the indices for parameter u, the middle 
  // index runs over v and the innermost vector index runs from 0...2 giving a 3-vector containing 
  // x, y, z coordinates for each point:
  vector<vector<vector<double>>> d;              // doubly nested vector of data
  d.resize(Nu);                                  // we have Nu blocks of data
  for(int i = 0; i < Nu; i++)                    // loop over the data blocks
  {
    d[i].resize(Nv);                             // each block has Nv lines/datapoints
    for(int j = 0; j < Nv; j++)                  // loop over lines in current block
    {
      d[i][j].resize(3);                         // each datapoint has 3 columns/dimensions
      d[i][j][0] = cos(u[i]) * (R+r*cos(v[j]));  // x = cos(u)*(R+r*cos(v))
      d[i][j][1] = sin(u[i]) * (R+r*cos(v[j]));  // y = sin(u)*(R+r*cos(v))
      d[i][j][2] = sin(v[j]) * r;                // z = sin(v)*r
    }
  }

  // plot:
  GNUPlotter p;                                  // create plotter object
  p.addData(d);                                  // pass the data to the plotter         
  p.addCommand("set hidden3d");                  // don't draw hidden lines
  p.addCommand("set view 20,50");                // set up perspective
  p.addCommand("set lmargin 0");                 // margin between plot and left border
  p.addCommand("set tmargin 0");                 // margin between plot and top border
  p.addCommand("set ztics 0.5");                 // density of z-axis tics
  p.plot3D();                                    // invoke GNUPlot

  //p.addCommand("set view equal xyz");  // otherwise, it stretches along z

  // ToDo: find out, why it is shown so small
  // ToDo: maybe plot 2 linked tori with different line colors  - that needs a more general formula 
  // for the torus where we can specify center and orientation - maybe have function 
  // createTorusData and a function affineTransform3D
}

// see here for a paramtric torus:
//https://en.wikipedia.org/wiki/Parametric_equation#3D_examples

// here are some more interesting shapes:
//http://soukoreff.com/gnuplot/

// here ar some other interesting examples that show contour plots, heatmaps, etc.
//https://www.packtpub.com/books/content/3d-plot-using-gnuplot


void demoHenneberg()
{
  // a Henneberg surface - see:
  // https://en.wikipedia.org/wiki/Henneberg_surface
  // http://mathworld.wolfram.com/HennebergsMinimalSurface.html
  // the data generation code has the same structure as in demoTorus, so we don't need comments 
  // here
  // todo: maybe the common stuff can be factored out to a function that takes 3 bivariate
  // std::function objects as (reference) parameters for the two functions x(u,v), y(u,v), z(u,v)
  // and Nu, Nv, uMin, uMax, vMin, vMax for the grid generation such that only have to call
  // plt.addParametricSurface(...) with lambda functions - but before implementing that, we should 
  // drag over the latest  changes to GNUPlotter.h/cpp from the RS-MET codebase
  // then we may also have addParametricCurve2D, addParametricCurve3D and add some more fun
  // surfaces like Moebius strip, the trefoil knot ...but for that, we need some way to thicken
  // a 3D curve into a solid object - maybe we need to estimate the tangent, normal and binormal
  // vector - maybe write an addThickCurve3D function that generates an appropriate mesh

  // user parameters:
  static const int Nu = 41;
  static const int Nv = 41;

  // generate data:
  double uu[Nu], vv[Nv];
  GNUPlotter::rangeLinear(uu, Nu, 0.0, 2*M_PI);
  GNUPlotter::rangeLinear(vv, Nv, 0.0, 2*M_PI);
  vector<vector<vector<double>>> d;
  d.resize(Nu);
  for(int i = 0; i < Nu; i++) {
    d[i].resize(Nv);
    for(int j = 0; j < Nv; j++) {
      double u = uu[i];
      double v = vv[j];
      d[i][j].resize(3);
      d[i][j][0] = 2*cos(v)*sinh(u) - (2./3)*cos(3*v)*sinh(3*u);
      d[i][j][1] = 2*sin(v)*sinh(u) + (2./3)*sin(3*v)*sinh(3*u);
      d[i][j][2] = 2*cos(2*v)*cosh(2*u); }}

  // plot:
  GNUPlotter p;
  p.addData(d);
  p.setPixelSize(800, 600);
  p.addCommand("set hidden3d");
  p.addCommand("set view 75,145");
  p.addCommand("set lmargin 0");
  p.addCommand("set tmargin 0");
  p.plot3D();   
}


void demoGaussianBivariate()
{
  // Demonstrates, how to plot a scalar function of 2 variables z = f(x,y) where the function value
  // z is interpreted as a height above an xy-plane. As example, a 2-dimensional (bivaritae) 
  // Gaussian distribution is used.

  GNUPlotter p;
  p.addCommand("set hidden3d");
  p.addCommand("set ztics 0.2");
  p.setGraphColors("000000", "000000"); // paint both sides black
  p.plotBivariateFunction(31, -2.0, 2.0, 31, -2.0, 2.0, &gauss2D);
}

void demoPow()
{
  GNUPlotter p;
  p.addCommand("set hidden3d");
  p.setAxisLabels("x", "y", "z=x^y");
  p.plotBivariateFunction(31, 0.0, 3.0, 31, 0.0, 3.0, &pow);
}

void demoSincRadial()
{
  // Demonstrates, how to plot a scalar function of 2 variables z = f(x,y) where the function value
  // z is interpreted as a height above an xy-plane. As example, a 2-dimensional (radial) sinc 
  // function is used.

  GNUPlotter p;
  p.addCommand("set hidden3d");
  p.addCommand("set view 45,35");
  p.setGraphColors("000000", "000000");
  p.plotBivariateFunction(51, -5.0, 5.0, 51, -5.0, 5.0, &sincRadial);

  // ToDo: check out how to do contour lines (or maybe fills), etc.
}

void demoSincRadialHeatMap()
{
  GNUPlotter p;
  p.addDataBivariateFunction(101, -5.0, 5.0, 101, -5.0, 5.0, &sincRadial);
  p.setPixelSize(450, 400);
  p.addCommand("set size square");                      // set aspect ratio to 1:1
  p.addGraph("i 0 nonuniform matrix w image notitle");   
  //p.addCommand("set palette color");                  // this is used by default
  //p.addCommand("set palette color negative");         // reversed colors
  //p.addCommand("set palette gray");                   // maximum is white
  //p.addCommand("set palette gray negative");          // maximum is black
  p.addCommand("set palette rgbformulae 30,31,32");     // colors printable as grayscale
  p.plot();

  // see http://gnuplot.sourceforge.net/demo_5.1/pm3d.html
  // for more examples of palettes
  // todo: add a convenience command setColorMap to GNUPlotter class

  // http://gnuplot.sourceforge.net/docs_4.2/node216.html lists the available rgbformulae
  // 7,5,15   ... traditional pm3d (black-blue-red-yellow)
  // 3,11,6   ... green-red-violet
  // 23,28,3  ... ocean (green-blue-white); try also all other permutations
  // 21,22,23 ... hot (black-red-yellow-white)
  // 30,31,32 ... color printable on gray (black-blue-violet-yellow-white)
  // 33,13,10 ... rainbow (blue-green-yellow-red)
  // 34,35,36 ... AFM hot (black-red-yellow-white)
  // 3,2,2    ... red-yellow-green-cyan-blue-magenta-red A full color palette in HSV color space 
}

void demoPendulumPhasePortrait()
{
  // physical parameters:
  double mu = 0.15; // damping constant
  double g  = 1;    // gravitational pull/acceleration
  double L  = 1;    // length of pendulum
  // see video by 3blue1brown video about this sort of plot:
  // https://www.youtube.com/watch?v=p_di4Zn4wz4 

  // two bivariate functions fx(x,y), fy(x,y) define the vector field:
  std::function<double(double, double)> fx, fy; 
  fx = []  (double x, double y) { return y; };
  fy = [&] (double x, double y) { return -mu*y - (g/L)*sin(x); };

  // create plotter and add graphs:
  GNUPlotter plt;
  plt.addVectorField2D(fx, fy, 51, -10., +10., 41, -4., +4.);  // vector field arrows
  plt.addFieldLine2D(  fx, fy, -9.9,  4.0, 0.1, 1000, 10);     // trajectory into right vortex
  plt.addFieldLine2D(  fx, fy, -4.0,  1.5, 0.1, 1000, 10);     // trajectory into middle vortex
  plt.addFieldLine2D(  fx, fy,  5.0, -3.0, 0.1, 1000, 10);     // trajectory into left vortex

  // setup plotting options and plot:
  plt.setTitle("Phase portrait of damped pendulum with 3 trajectories"); 
  plt.addCommand("set palette rgbformulae 30,31,32 negative"); // arrow color-map
  plt.setGraphColors("209050");                                // trajectory color
  plt.addCommand("set xrange [-10.5:10.5]");
  plt.addCommand("set yrange [-4.5:4.5]");
  plt.addCommand("set xlabel \"Angle {/Symbol q}\"");
  plt.addCommand("set ylabel \"Angular velocity {/Symbol w}\"");
  plt.addCommand("set xtics pi");
  plt.addCommand("set format x '%.0P{/Symbol p}'");
  plt.setPixelSize(1000, 500); 
  plt.plot();

  // additional info:
  // greek and/or latex letters:
  // https://sourceforge.net/p/gnuplot/discussion/5925/thread/bc8a65fe/
  // http://www.gnuplot.info/files/tutorial.pdf
  // https://tex.stackexchange.com/questions/119518/how-can-add-some-latex-eq-or-symbol-in-gnuplot
  // https://stackoverflow.com/questions/28964500/math-in-axes-labels-for-gnuplot-epslatex-terminal-not-formatted-correctly

  // tics at multiples of pi:
  // http://www.gnuplotting.org/tag/tics/  // ...it also says something about multiplots
  // http://www.gnuplotting.org/set-your-tic-labels-to-use-pi-or-to-be-blank/
}





// maybe move to experiments:
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

std::complex<double> dipoleField(std::complex<double> z,
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
void demoDipole()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return dipoleField(z); };
  GNUPlotter::plotComplexVectorField(f, 41, -2., +2.0, 21, -1.0, +1.0, false);
}








// here is a good tutorial:
// http://lowrank.net/gnuplot/index-e.html

// demo for various line styles:
// http://gnuplot.sourceforge.net/demo_4.6/dashcolor.html

// for polar grids
// http://stackoverflow.com/questions/6772135/how-to-get-a-radialpolar-plot-using-gnu-plot

// for 3D plot, manual, pages 65,115,184
// http://lowrank.net/gnuplot/datafile-e.html

// for complex functions:
// http://gnuplot.sourceforge.net/demo/complex_trig.html

// heatmaps:
// http://gnuplot.sourceforge.net/demo_5.1/pm3d.html
// http://www.gnuplotting.org/tag/pm3d/ -> plot 'heat_map_data.txt' matrix with image
// http://www.kleerekoper.co.uk/2014/05/how-to-create-heatmap-in-gnuplot.html
// http://skuld.bmsc.washington.edu/~merritt/gnuplot/demo_canvas/heatmaps.html

// for more demo/example plots: 
// -Möbius Strip
//  https://en.wikipedia.org/wiki/M%C3%B6bius_strip
//
// -Integer plot: fibonacci sequence (or prime numbers) - maybe we can have a special int plot
//  that writes the values above the datapoints (centered) - but maybe this is something for a 
//  subclass that deals specifically with integer plots
// -Gaussian bells with diffent mu, sigma - also demonstrates how to use greek letters
// -square and/or saw wave with linear, cubic and sinc interploation - use points and impulses for
//  the discrete time data, lines of different colors for the interpolants
// -pole/zero plot of an elliptic bandpass in the z-domain
// -for log/log plots: familiy of Butterworth magnitude responses
//  -maybe with phase-plots in the same plot (angles written on right y-axis, we need an expression 
//   for the (unwrapped) phase
// -Bode Plot: http://gnuplot.sourceforge.net/demo/multiaxis.html
// -3D: 2D Gaussian, z- or s-domain pole/zero "landscapes", clouds of points (maybe using gaussian
//  distributions)
// -spectrogram-plots, phasogram-plots
// -histogram: use equal distributions for HLS values, convert to RGB and plot histogram of RGB values


  // move to function testPointTypes
  //plt.setGraphStyles(5, "points pt 1", "points pt 2", "points pt 3", "points pt 4", "points pt 5");
  //plt.setGraphStyles(5, "points pt 6", "points pt 7", "points pt 8", "points pt 9", "points pt 10");

  // pointtypes: 1: +, 2: crosses, 3: *, 4: squares, 5: filled squares, 6: circles, 
  // 7: filled circles, 8: triangle up, 9: filled triangle up, 10: triangle down,

  // move to function testLineStyles:
  //plt.setGraphStyles(4, "lines", "lines lw 2", "points pt 6 ps 2", "linespoints pt 7 ps 1");
  //plt.setGraphStyles(5, "lines", "points pt 2", "impulses", "boxes", "linespoints pt 7 ps 1");
  //plt.setGraphStyles(4, "lines lt 1", "lines lt 2", "lines lt 3", "lines lt 3");
    // should create dashed lines but doesn't work






