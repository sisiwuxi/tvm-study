// #include<SDL.h>
#include<GL/gl.h>
#include<GL/glext.h>
#include <stdio.h>
#include <cuda_runtime.h>
// #include <cutil_inline.h>
// #include <cutil_gl_inline.h>
// #include <cutil_gl_error.h>
// #include <rendercheck_gl.h>


GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

__global__ void createVertices(float4* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    // Write positions
    positions[y * width + x] = make_float4(u, w, v, 1.0f);
}

void display()
{
    const int height = 32;//1024;
    const int width = 32;//1024;
    // Map buffer object for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes,positionsVBO_CUDA);
    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    float time = 0.5;
    createVertices<<<dimGrid, dimBlock>>>(positions, time, width, height);
    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glBindTexture(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);
    // Swap buffers
    // glutSwapBuffers();
    // glutPostRedisplay();
}

void deleteVBO()
{
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    // glDeleteBuffers(1, &positionsVBO);
    glDeleteTextures(1, &positionsVBO);
}


// CUTBoolean initGL(int argc, char **argv)
GLboolean initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop Demo (adapted from NVDIA's simpleGL)");
 
	glutDisplayFunc(fpsDisplay);
 
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0"))
	{
		fprintf(stderr, "ERROR: Support for necessary OpengGL extensions missing.");
		return CUTFalse;
	}
 
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
 
	glViewport(0, 0, window_width, window_height);
 
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);
 
	return CUTTrue;
}

int main()
{
    // Initialize OpenGL and GLUT for device 0
    // and make the OpenGL context current
    initGL();
    glutDisplayFunc(display);
    // Explicitly set device 0
    cudaSetDevice(0);
    // Create buffer object and register it with CUDA
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = width * height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
    positionsVBO,
    cudaGraphicsMapFlagsWriteDiscard);
    // Launch rendering loop
    glutMainLoop();
    ...
}
