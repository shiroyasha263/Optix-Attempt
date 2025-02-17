#pragma once

#include "optix7.h"

template <typename T>
struct StructuredBuffer {
	T* data;
	size_t size;
};

struct TriangleMeshSBTData {
	float3 color;
	float3 emmisive;
	float3 specular;
	float shininess;
	float ior;
	enum MaterialType {
		DIFFUSE,
		SPECULAR,
		DIELECTRIC
	} matType;
	StructuredBuffer<float3> vertex;
	StructuredBuffer<float3> normal;
	StructuredBuffer<float2> texcoord;
	StructuredBuffer<int3> index;
	bool hasTexture;
	CUtexObject texture;
};

struct LaunchParams {
	StructuredBuffer<uint32_t> colorBuffer;
	StructuredBuffer<float3> accumBuffer;
	int2 fbSize;

	struct Camera {
		float3 position;
		float3 direction;
		float3 horizontal;
		float3 vertical;
	} camera;

	OptixTraversableHandle traversable;
	unsigned int frameID{ 0 };
};