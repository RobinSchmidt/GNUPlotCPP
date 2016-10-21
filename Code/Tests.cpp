#include "Tests.h"

void testDataSetup()
{
  // 2 columns, 6 rows separated into 3 blocks of lengths 2,1,3. 
  int data[2][6]      = {{1,3,5,7,9,11},{2,4,6,8,10,12}};
  int blockLengths[3] = {2,1,3};
  int *pData[2];
  pData[0] = data[0];
  pData[1] = data[1];

  GNUPlotter p;
  p.addData(3, blockLengths, 2, pData);
  p.addData(6, 2, pData);                // should write the same data without spaces  

  // the datafile should look like:
  //1 2
  //3 4
  //
  //5 6
  //
  //7 8
  //9 10
  //11 12
  //
  //
  //1 2
  //3 4
  //5 6
  //7 8
  //9 10
  //11 12
  //
  //
}

void testDataSetup2()
{
  int x[3] = { 1,2,3 };
  int y[3] = { 4,5,6 };
  int z[3] = { 7,8,9 };
  GNUPlotter p;
  p.addDataArrays(3, x, y, z);

  // the datafile should look like:
  //1 4 7
  //2 5 8
  //3 6 9
  //
  //
}

void testMatrixWrite()
{
  // the positions of the gridlines:
  int x[4] = { 1,2,3,4 };
  int y[3] = { 2,4,8 };

  // the dataset as matrix where the 1st index runs over the x-gridlines and the 2nd over the 
  // y-gridlines:
  int z[4][3] ={ {21,41,81},{22,42,82},{23,43,83},{24,44,84} };
  int *pz[4];
  for(int i = 0; i < 4; i++)
    pz[i] = z[i];

  // create plotter, add dataset 2 times, once as matrix and once as (x,y,z)-triplets and plot once
  // with lines and once with points
  GNUPlotter p;
  p.addDataMatrix(4, 3, x, y, pz);
  p.addDataGrid(4, 3, x, y, pz);
  p.setAxisLabels("x", "y", "z");
  p.addCommand("set hidden3d");
  p.addGraph("i 0 nonuniform matrix w lines notitle");
  p.addGraph("i 1 w points pt 7 ps 1.2 notitle");
  p.plot3D();

  // the datafile should look like:
  //4  1  2  3  4
  //2 21 21 23 24
  //4 41 42 43 44
  //8 81 82 83 84
  //
  //
  //1 2 21
  //1 4 41
  //1 8 81
  //
  //2 2 22
  //2 4 42
  //2 8 82
  //
  //3 2 23
  //3 4 43
  //3 8 83
  //
  //4 2 24
  //4 4 44
  //4 8 84
  //
  //
}

void testLowLevelCalls()
{
  // We add all commands including the actual 'plot' command manually without using any convenience
  // functions
  // todo: make a function demoDashPatterns

  GNUPlotter p;

  // add data to the datafile:
  p.addDataFunctions(201, 0.0, 10.0, &sin, &cos);
  p.addDataFunctions(21,  0.0, 10.0, &sin, &cos);

  // here's information how to set the dash-types:
  // http://stackoverflow.com/questions/19412382/gnuplot-line-types

  // define linetypes:
  p.addCommand("set lt 1 lc rgb \"#800000\" lw 1.5 dt (2,8)");
  p.addCommand("set lt 2 lc rgb \"#800000\" pt 7 ps 1.2");
  p.addCommand("set lt 3 lc rgb \"#006000\" lw 1.5 dt (1,8,5,8)");
  p.addCommand("set lt 4 lc rgb \"#006000\" pt 7 ps 1.2");
  // shorthand syntax: lt: linetype, lw: linewidth, dt: dashtype, pt: pointtype, ps: pointsize
  // dashtype (s1,e1,s2,e2,s3,e3,s4,e4) 
  // # dash pattern specified by 1 to 4
  // # numerical pairs <solid length>, <emptyspace length>

  // test:
  //p.addCommand("set lt 3 lc rgb \"#000080\"");
  //p.addCommand("set lt 3 dt (1,0)");


  // test: set linetype cycle 5
  //p.addCommand("set linetype cycle 1"); // doesn't seem to work


  // create and add the plot command to the commandfile:
  string s;
  string dp = p.getDataPath();
  s += "plot \\\n";
  s += "'" + dp + "' i 0 u 1:2 w lines lt 1 notitle,\\\n";
  s += "'" + dp + "' i 1 u 1:2 w points lt 2 notitle,\\\n";
  s += "'" + dp + "' i 0 u 1:3 w lines lt 3 notitle,\\\n";
  s += "'" + dp + "' i 1 u 1:3 w points lt 4 notitle";
  p.addCommand(s);
  // shorthand syntax: i: index, u: using, w: with



  // invoke GNUPlot:
  p.invokeGNUPlot();


  int dummy = 0;
}