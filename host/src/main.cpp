#define PROGRAM_FILE "fft.cl"
#define KERNEL_FUNC "fft"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <iostream>
//fstream
#include "CL/opencl.h"
#include "AOCL_Utils.h"

using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;

// Function prototypes
bool init_opencl();
bool cleanup();

// Entry point.
int main() {

//******************************************************************************
// Initialize OpenCL
//******************************************************************************

  if(!init_opencl())  return -1;

  cl_int status;

//******************************************************************************
// Load values
//******************************************************************************

  //Number of points
  uint N=32;
  float data[N*2];
  float fftData[N*2];

  //srand(time(NULL));
  for(int i=0; i<(N); i++) {                         //32 times but data is 64
    data[2*i] = i+1;
    data[2*i+1] = 0;
  }


  cl_command_queue Queue = NULL;
  cl_kernel Kernel = NULL;
  cl_mem input_buffer;
  cl_mem output_buffer;

  cl_ulong local_mem_size;

  // Create command queue.
  Queue = clCreateCommandQueue(context, device,
              CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create queue");

  // Create kernels
  Kernel = clCreateKernel(program, KERNEL_FUNC, &status);
  checkError(status, "Failed to create kernel");

  // Create buffer data
  input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                2*N*sizeof(float), NULL, &status);
  checkError(status, "Failed to create input buffer");

  //Create output buffer
  output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                   2*N*sizeof(float), NULL, &status);
  checkError(status, "Failed to create output buffer");

  status = clEnqueueWriteBuffer(Queue, input_buffer, CL_TRUE, 0,
            2*N*sizeof(float), data, 0, NULL, NULL);
  checkError(status, "Failed to transfer data");

  local_mem_size = 2*N*sizeof(float);

  status = clSetKernelArg(Kernel, 0, sizeof(cl_mem), &input_buffer);
  status |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), &output_buffer);
  status |= clSetKernelArg(Kernel, 2, local_mem_size, NULL);
  status |= clSetKernelArg(Kernel, 3, sizeof(uint), &N);

  checkError(status, "Couldn't set kernel arguments");

  std::cout << "/* Launching kernel" << '\n';

  status = clEnqueueTask(Queue, Kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");


  status = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE,
    0, 2*N*sizeof(float), &fftData, 0, NULL, NULL);
  checkError(status, "Failed to read result");

  for(int k=0;k<2*N;k++) printf("%i %f \n",k + 1,fftData[k]);

  /* Deallocate resources */
  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseKernel(Kernel);
  clReleaseCommandQueue(Queue);

  if(!cleanup())  return -1;
  return 0;
}

/*******************************************************************************
/*
/* OpenCL
/*
/*******************************************************************************/


bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.
  cl_uint num_devices;
  scoped_array<cl_device_id> devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0];

  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(devices[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program.
  std::string binary_file = getBoardBinaryFile(KERNEL_FUNC, devices[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context,
          binary_file.c_str(), devices, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return true;
}

bool cleanup(){

  /* Deallocate resources */

  clReleaseProgram(program);
  clReleaseContext(context);

  return true;

}
