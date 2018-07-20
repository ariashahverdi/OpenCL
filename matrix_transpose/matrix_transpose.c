// tranposeMatrix.c
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/cl.h>

#define PROGRAM_FILE "transposeMatrix.cl"
#define KERNEL_FUNC "matrixTranspose"

#define DATA_SIZE 65536
#define ROWS 16
#define COLS 16
#define MATSIZE ROWS * COLS

/* Find a GPU or CPU associated with the first available platform */

cl_device_id create_device() {
    
    cl_platform_id platform;
    cl_device_id device_id;
    int err;
    
    // Identify a platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }
    
    // Access a device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    }
    if(err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }
    
    // Get the name of the device
    size_t returned_size = 0;
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    if (err != CL_SUCCESS)     {
        printf("Error: Failed to get device Info!\n");
        exit(1);
    }
    printf("Connecting to %s ...\n", device_name);
    
    return device_id;
}


/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;
    
    // Read program file and place content into buffer
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    
    // Create program from file
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    
    // Build program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        
        // Find size of log and print to std output
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    
    return program;
}


int main()
{
    cl_float *inputMatrix;
    cl_float *results;
    cl_uint width = COLS;
    cl_uint height = ROWS;

    // OpenCL host variables go here:
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_int i, j, err;
    size_t local_size, global_size;
    
    // Data and buffers
    cl_mem input_buffer, output_buffer;
    size_t global[2];
    cl_int num_groups;

    // variables used to read kernel source file
    FILE *fp;
    long filelen;
    long readlen;
    char *kernel_src;  // char string to hold kernel source
   
    // initialize inputMatrix with some data and print it
    int x, y;
    int data=0;
    inputMatrix = malloc(sizeof(cl_float)*width*height);
    results = malloc(sizeof(cl_float)*width*height);
    printf("Input Matrix\n");
    for(y=0;y<height;y++)
    {
        for(x=0;x<width;x++)
        {
            inputMatrix[y*height+x]= data;
            results[y*height+x]=0;
            data++;
            printf("%.2f, ",inputMatrix[y*height+x]);
        }
        printf("\n");
    }
    
    /*
    // read the kernel
        fp = fopen("transposeMatrix_kernel.cl","r");
        fseek(fp,0L, SEEK_END);
        filelen = ftell(fp);
        rewind(fp);
        kernel_src = malloc(sizeof(char)*(filelen+1));
        readlen = fread(kernel_src,1,filelen,fp);
        if(readlen!= filelen)
        {
            printf("error reading file\n");
            exit(1);
        }

    // ensure the string is NULL terminated
        kernel_src[filelen+1]='\0';
    */

    // ----- Insert OpenCL host source here ----
    
    // Create device and context
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context)
    {
        perror("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if(err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    // Build program
    program = build_program(context, device, PROGRAM_FILE);

    // Create data buffer
    global_size = MATSIZE;
    local_size = 4;
    num_groups = global_size/local_size;
    input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                  CL_MEM_COPY_HOST_PTR, MATSIZE * sizeof(cl_float), inputMatrix, &err);
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                                  CL_MEM_COPY_HOST_PTR, MATSIZE * sizeof(cl_float), results, &err);
    if(err < 0) {
        perror("Couldn't create a buffer");
        exit(1);
    };
    
    // Create a kernel
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    };
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
    if(err < 0) {
        perror("Couldn't create a kernel argument");
        exit(1);
    }
    
    global[0] = width;
    global[1] = height;
    
    // Enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global,
                                 NULL, 0, NULL, NULL);
    if(err < 0) {
        perror("Couldn't enqueue the kernel");
        exit(1);
    }
    
    // wait for the command to finish
    clFinish(queue);
    
    // Read the kernel's output
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                              sizeof(cl_float)*width*height, results, 0, NULL, NULL);
    if(err < 0) {
        perror("Couldn't read the buffer");
        exit(1);
    }

    printf("Transposed Matrix\n");
    for(y=0;y<height;y++)
    {
        for(x=0;x<width;x++)
        {
            printf("%.2f, ",results[y*height+x]);
        }
        printf("\n");
    }
    
    // Deallocate resources
    clReleaseKernel(kernel);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    
    // ----- End of OpenCL host section ----


    
    
    return 0;
}
