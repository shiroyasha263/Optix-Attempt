#include "SampleRenderer.h"

#include <optix_function_table_definition.h>

extern "C" char embedded_ptx_code[];

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayGenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// This is my uniform data that is being passed to the shader!!
	TriangleMeshSBTData data;
};

SampleRenderer::SampleRenderer(const Model* model) : model(model) {
	initOptix();

	std::cout << "Optix Renderer: Creating Optix context ..\n";
	createContext();

	std::cout << "Optix Renderer: Setting up module ..\n";
	createModule();

	std::cout << "Optix Renderer: Creating raygen programs ..\n";
	createRaygenPrograms();

	std::cout << "Optix Renderer: Creating miss programs ..\n";
	createMissPrograms();

	std::cout << "Optix Renderer: Creating hitgroup programs ..\n";
	createHitgroupPrograms();

	std::cout << "Optix Renderer: Creating Acceleration Structure ..\n";
	launchParams.traversable = buildAccel();

	std::cout << "Optix Renderer: Setting up optix pipeline ..\n";
	createPipeline();

	std::cout << "Optix Renderer: Creating textures to pass ..\n";
	createTextures();

	std::cout << "Optix Renderer: Building shader binding table ..\n";
	buildSBT();

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "Optix Renderer: Everything finally set up.. \n";

	std::cout << TERMINAL_GREEN << "Optix Renderer: Ready to be used \n" << TERMINAL_DEFAULT;
}

void SampleRenderer::createTextures() {
	int numTextures = (int)model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);
    
    for (int textureID=0;textureID<numTextures;textureID++) {
      auto texture = model->textures[textureID];
      
      cudaResourceDesc res_desc = {};
      
      cudaChannelFormatDesc channel_desc;
      int32_t width  = texture->resolution.x;
      int32_t height = texture->resolution.y;
      int32_t numComponents = 4;
      int32_t pitch  = width*numComponents*sizeof(uint8_t);
      channel_desc = cudaCreateChannelDesc<uchar4>();
      
      cudaArray_t   &pixelArray = textureArrays[textureID];
      CUDA_CHECK(cudaMallocArray(&pixelArray,
                             &channel_desc,
                             width,height));
      
      CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
                                 /* offset */0,0,
                                 texture->pixel,
                                 pitch,pitch,height,
                                 cudaMemcpyHostToDevice));
      
      res_desc.resType          = cudaResourceTypeArray;
      res_desc.res.array.array  = pixelArray;
      
      cudaTextureDesc tex_desc     = {};
      tex_desc.addressMode[0]      = cudaAddressModeWrap;
      tex_desc.addressMode[1]      = cudaAddressModeWrap;
      tex_desc.filterMode          = cudaFilterModeLinear;
      tex_desc.readMode            = cudaReadModeNormalizedFloat;
      tex_desc.normalizedCoords    = 1;
      tex_desc.maxAnisotropy       = 1;
      tex_desc.maxMipmapLevelClamp = 99;
      tex_desc.minMipmapLevelClamp = 0;
      tex_desc.mipmapFilterMode    = cudaFilterModePoint;
      tex_desc.borderColor[0]      = 1.0f;
      tex_desc.sRGB                = 0;
      
      // Create texture object
      cudaTextureObject_t cuda_tex = 0;
      CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
      textureObjects[textureID] = cuda_tex;
	}
}

OptixTraversableHandle SampleRenderer::buildAccel() {

	const int numMeshes = (int)model->meshes.size();
	
	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);

	OptixTraversableHandle asHandle{ 0 };

	// ==================================================================
	// triangle inputs
	// ==================================================================
	std::vector<OptixBuildInput> triangleInput(numMeshes);
	// create local variables, because we need a *pointer* to the
	// device pointers
	std::vector<CUdeviceptr> d_vertices(numMeshes);
	std::vector<CUdeviceptr> d_indices(numMeshes);
	std::vector<uint32_t> triangleInputFlags(numMeshes);

	for (int meshID = 0; meshID < numMeshes; meshID++) {
		
		TriangleMesh& mesh = *model->meshes[meshID];
		vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
		indexBuffer[meshID].alloc_and_upload(mesh.index);
		if (!mesh.normal.empty())
			normalBuffer[meshID].alloc_and_upload(mesh.normal);
		if (!mesh.texcoord.empty())
			texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

		triangleInput[meshID] = {};
		triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
		d_indices[meshID] = indexBuffer[meshID].d_pointer();

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
		triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];
					 
		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];
		
		triangleInputFlags[meshID] = 0;

		// Update Needed!!!!!!!!
		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;

	}
      
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)numMeshes,  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)numMeshes,
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}


void SampleRenderer::initOptix() {
	cudaFree(0);

	int numDevices;
	// The name is pretty self explanatory
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("No CUDA capable devices found\n");

	std::cout << "Optix Renderer: Found " << numDevices << " CUDA capable devices\n";

	OPTIX_CHECK(optixInit());

	std::cout << TERMINAL_GREEN
		<< "Optix Renderer: Optix Successfully Initialized!!\n"
		<< TERMINAL_DEFAULT;
}

static void context_log_cb(	unsigned int level,
							const char* tag,
							const char* message,
							void*) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void SampleRenderer::createContext() {
	
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "Optix Renderer: Running on device " << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
	to use. in this simple example, we use a single module from a
	single .cu file, using a single embedded ptx string */
void SampleRenderer::createModule() {

	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	// FOCUS HERE WHEN U GET SOME BUG REGARDING ACCEL STRUCTURE!!!!!!!
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// FOCUS HERE WHEN U GET SOME BUG REGARDING ACCEL STRUCTURE!!!!!!!
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "SLANG_globalParams";

	pipelineLinkOptions.maxTraceDepth = 2;

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreate(	optixContext,
									&moduleCompileOptions, 
									&pipelineCompileOptions,
									ptxCode.c_str(),
									ptxCode.size(),
									log,
									&sizeof_log,
									&module));

	if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::createRaygenPrograms() {

	raygenPGs.resize(1);
	
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};

	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	// Where to find the program group code!
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&raygenPGs[0]));

	if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::createMissPrograms() {

	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};

	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	// Where to find the program group code!
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__miss_radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&missPGs[0]));

	if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::createHitgroupPrograms() {

	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};

	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	// Where to find the program group code!
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__closesthit_radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__anyhit_radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&hitgroupPGs[0]));

	if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::createPipeline() {
	std::vector<OptixProgramGroup> programGroups;

	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext,
									&pipelineCompileOptions,
									&pipelineLinkOptions,
									programGroups.data(),
									(int)programGroups.size(),
									log, &sizeof_log,
									&pipeline));

	if (sizeof_log > 1) PRINT(log);

	// Maybe look into this if got time
	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		pipeline,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));

	if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::buildSBT() {
	// ------------------------------------------------------------------
	// build raygen records
	// ------------------------------------------------------------------
	std::vector<RayGenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RayGenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

	// we don't actually have any objects in this example, but let's
	// create a dummy one so the SBT doesn't have any null pointers
	// (which the sanity checks in compilation would complain about)
	int numObjects = (int)model->meshes.size();
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshID = 0; meshID < numObjects; meshID++) {
		auto mesh = model->meshes[meshID];

		HitgroupRecord rec;
		// Hitgroup PG is what tells it to approach which hitgroup program
		// If the hit model/mesh/object is a glass pass it to glass hitgroupPGs
		// This is what my understanding is
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));

		rec.data.color = *((float3*)(&mesh->diffuse));
		if (mesh->diffuseTextureID >= 0) {
			rec.data.hasTexture = true;
			rec.data.texture = textureObjects[mesh->diffuseTextureID];
		}
		else {
			rec.data.hasTexture = false;
		}
		rec.data.vertex.data = (float3*)vertexBuffer[meshID].d_pointer();
		rec.data.vertex.size = vertexBuffer[meshID].sizeInBytes / sizeof(float3);
		rec.data.normal.data = (float3*)normalBuffer[meshID].d_pointer();
		rec.data.normal.size = normalBuffer[meshID].sizeInBytes / sizeof(float3);
		rec.data.texcoord.data = (float2*)texcoordBuffer[meshID].d_pointer();
		rec.data.texcoord.size = texcoordBuffer[meshID].sizeInBytes / sizeof(float2);
		rec.data.index.data = (int3*)indexBuffer[meshID].d_pointer();
		rec.data.index.size = indexBuffer[meshID].sizeInBytes / sizeof(float3);
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void SampleRenderer::render() {
	// Just a check to make sure there is a pixel to draw
	if (launchParams.fbSize.x == 0) return;
	
	launchParamsBuffer.upload(&launchParams, 1);

	OPTIX_CHECK(optixLaunch(pipeline,	// The pipeline we want to launch
							stream,		// The stream we want to launch it on
										// The parameters we want to pass
							launchParamsBuffer.d_pointer(),
							launchParamsBuffer.sizeInBytes,
							&sbt,		// The shader table we want it to use
							launchParams.fbSize.x, 
							launchParams.fbSize.y,
							1));

	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	CUDA_SYNC_CHECK();
	launchParams.frameID++;
}

void SampleRenderer::resize(const glm::ivec2& newSize) {
	if (newSize.x == 0 | newSize.y == 0) return;

	colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

	launchParams.fbSize = int2{ newSize.x, newSize.y };
	launchParams.colorBuffer.data = (uint32_t*)colorBuffer.d_ptr;
	launchParams.colorBuffer.size = newSize.x * newSize.y;
	launchParams.frameID = 0;
}

void SampleRenderer::setCamera(const Camera& camera) {
	glm::vec3 pos = camera.from;
	glm::vec3 dir = glm::normalize(camera.at - camera.from);
	lastSetCamera = camera;
	launchParams.camera.position = *((float3*)(&pos));
	launchParams.camera.direction = *((float3*)(&dir));
	const float cosFovy = 0.66f;
	const float aspect = launchParams.fbSize.x / float(launchParams.fbSize.y);
	glm::vec3 horizontal = cosFovy * aspect * glm::normalize(glm::cross(dir, camera.up));
	glm::vec3 vertical = cosFovy * glm::normalize(glm::cross(horizontal, dir));
	launchParams.camera.horizontal = *((float3*)(&horizontal));
	launchParams.camera.vertical = *((float3*)(&vertical));
	launchParams.frameID = 0;
}

void SampleRenderer::downloadPixels(uint32_t h_pixels[]) {
	colorBuffer.download(h_pixels,
						 launchParams.fbSize.x * launchParams.fbSize.y);
}