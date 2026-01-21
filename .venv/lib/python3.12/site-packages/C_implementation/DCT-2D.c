#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "args.h"
#include "mallok.h"
#include "DCT.h"

#define ROWS 512
#define COLS 512

int main(int argc, char *argv[]) {
  if(exist_arg((char *)"-help",argc,argv)) {
    fprintf(stderr,"Compute the 2D-DCT\n");
    fprintf(stderr," [-inverse] (inverse transform)\n");
    fprintf(stderr," [-input <(unsigned char) image file>] (default stdin)\n");
    fprintf(stderr," [-output <(double) image file>] (default stdout)\n");
    fprintf(stderr," [-rows <number of rows>] (defaults %d)\n",ROWS);
    fprintf(stderr," [-cols <number of columns>] (default %d)\n",COLS);
  } else {
    clock_t time;

    FILE *input_file, *output_file;
    int rows = ROWS;
    int cols = COLS;
    double **data;
    int i,j;
    unsigned char *line;
   
    input_file = stdin;
    if(EXIST_ARG((char *)"-input")) {
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: input file = %s\n",argv[0],GET_ARG((char *)"-input"));
      }
      input_file = fopen(GET_ARG((char *)"-input"),"rb");
      if(!input_file) {
        fprintf(stderr,"%s: unable to open input file: %s\n",argv[0],GET_ARG((char *)"-input"));
        return 1;
      }
    } else {
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: input file = stdin\n",argv[0]);
      }
    }

    output_file= stdout;
    if(EXIST_ARG((char *)"-output")) {
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: output file = %s\n",argv[0],GET_ARG((char *)"-output"));
      }
      output_file = fopen(GET_ARG((char *)"-output"),"wb");
      if(!output_file) {
        fprintf(stderr,"%s: unable to open output file: %s\n",argv[0],GET_ARG((char *)"-output"));
        return 1;
      }
    } else {
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: output file = stdout\n",argv[0]);
      }
    }

    if(EXIST_ARG((char *)"-rows")) rows = atoi(GET_ARG((char *)"-rows"));
    if(EXIST_ARG((char *)"-cols")) cols = atoi(GET_ARG((char *)"-cols"));
    if(EXIST_ARG((char *)"-v")) {
      fprintf(stderr,"%s: rows = %d\n",argv[0],rows);
      fprintf(stderr,"%s: cols = %d\n",argv[0],cols);
    }

    line = (unsigned char *)malloc(cols*sizeof(unsigned char));
    data = (double **)malloc(rows,cols,sizeof(double));

    if(!EXIST_ARG((char *)"-inverse")) {

      fprintf(stderr, "%s: forward transform (%d,%d)\n",argv[0], rows, cols);

      time = clock();
      for(j=0;j<rows;j++) {
	fread(line,sizeof(unsigned char),cols,input_file);
	for(i=0; i<cols; i++) {
	  data[j][i] = (double)line[i];
	}
      }
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: input data time = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

      time = clock();
      fct2d(data,rows,cols);
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: transforming time = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

      time = clock();
      for(j=0;j<rows;j++) {
	fwrite(data[j],sizeof(double),cols,output_file);
      }
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: output data = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

    } else {

      fprintf(stderr, "%s: inverse transform\n",argv[0]);

      time = clock();
      for(j=0;j<rows;j++) {
	fread(data[j],sizeof(double),cols,input_file);
      }
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: input data time = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

      time = clock();
      ifct2d(data,rows,cols);
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: inverse transforming = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

      time = clock();
      for(j=0;j<rows;j++) {
	for(i=0; i<cols; i++) {
	  if(data[j][i] > 255) line[i] = 255;
	  else if(data[j][i] < 0) line[i] = 0;
	  else line[i] = (unsigned char)data[j][i];
	}
	fwrite(line,sizeof(unsigned char),cols,output_file);
      }
      time = clock() - time;
      if(EXIST_ARG((char *)"-v")) {
	fprintf(stderr,"%s: input data time = %.2f seconds\n",argv[0],(float)time/CLOCKS_PER_SEC);
      }

    }

  }
}
