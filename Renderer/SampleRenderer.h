#pragma once

#include <iostream>
#ifndef TERMINAL_COLORS
#define TERMINAL_COLORS
#define TERMINAL_RED "\033[1;31m"
#define TERMINAL_GREEN "\033[1;32m"
#define TERMINAL_YELLOW "\033[1;33m"
#define TERMINAL_BLUE "\033[1;34m"
#define TERMINAL_RESET "\033[0m"
#define TERMINAL_DEFAULT TERMINAL_RESET
#define TERMINAL_BOLD "\033[1;1m"
#endif
#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#endif

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

struct Camera {
	glm::vec3 from;
	glm::vec3 at;
	glm::vec3 up;
};

class SampleRenderer {
public:
	SampleRenderer(const Model *model);

	void render();
	
	void resize(const glm::ivec2 &newSize);

	void downloadPixels(uint32_t h_pixels[]);

	void setCamera(const Camera& camera);

protected:
	void initOptix();
	
	void createContext();

	void createModule();

	void createRaygenPrograms();

	void createMissPrograms();

	void createHitgroupPrograms();

	void createPipeline();

	void buildSBT();

	OptixTraversableHandle buildAccel();

protected:

	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixContext;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;

	OptixShaderBindingTable sbt = {};

	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;

	CUDABuffer colorBuffer;

	Camera lastSetCamera;

	const Model* model;
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> indexBuffer;
	//! buffer that keeps the (final, compacted) accel structure
	CUDABuffer asBuffer;
};