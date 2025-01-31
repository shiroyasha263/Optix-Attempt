struct Camera {
    float3 position;
    float3 direction;
    float3 horizontal;
    float3 vertical;
};

//------------------------------------------------------------------------------
// All global variables are stored in constant memory, under the
// "SLANG_globalParams" structure. These parameters are filled in
// by optix upon optixLaunch via launch parameters.
//
// Use the "-dump-intermediates" flag with the slang compiler to see
// the exact generated layout.
//------------------------------------------------------------------------------

// Launch Params

RWStructuredBuffer<uint> colorBuffer;
int2 fbSize;
Camera camera;
RaytracingAccelerationStructure traversable;

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

[shader("closesthit")]
void closesthit_radiance(
    inout float3 prd: SV_RayPayload, 
    in float2 barycentrics,
    uniform float3 color,
    uniform RWStructuredBuffer<float3> vertices,
    uniform RWStructuredBuffer<float3> normals,
    uniform RWStructuredBuffer<float2> texcoords,
    uniform RWStructuredBuffer<int3> indices,
    uniform bool hasTexture,
    uniform Texture2D texture) {

    const int primID = PrimitiveIndex();
    int3 index = indices[primID];
    const float u = barycentrics.x;
    const float v = barycentrics.y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    float3 N;
    uint numNormals, normalStride;
    normals.GetDimensions(numNormals, normalStride);
    
    // Shading normals being used
    if (numNormals > 0) {
        N = (1.f - u - v) * normals[index.x]
            + u * normals[index.y]
            + v * normals[index.z];
    } else { // Geometric normals being used
        const float3 A = vertices[index.x];
        const float3 B = vertices[index.y];
        const float3 C = vertices[index.z];
        N = normalize(cross(B - A, C - A));
    }

    float3 diffuseColor = color;
    uint numTexcoords, texcoordStride;
    texcoords.GetDimensions(numTexcoords, texcoordStride);
    if (hasTexture && numTexcoords > 0) {
        const float2 tc = (1.f - u - v) * texcoords[index.x]
                          + u * texcoords[index.y]
                          + v * texcoords[index.z];
        SamplerState temp;
        float4 fromTexture = texture.Sample(temp, tc);
        diffuseColor *= fromTexture.rgb;
    }

    float3 rayDir = WorldRayDirection();
    float cosDN = 0.2f + 0.8f * abs(dot(N, rayDir));
    prd = diffuseColor * 0.2f + cosDN * diffuseColor * 0.8f;
}

[shader("anyhit")]
void anyhit_radiance() {

    // For now empty
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

// inout I think means that it is both outputed by this shader and taken in as input
[shader("miss")]
void miss_radiance(inout float3 prd: SV_RayPayLoad) {
    // set to constant white as background color
    prd = float3(1.f, 1.f, 1.f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
[shader("raygeneration")]
void renderFrame() {
    const int ix = DispatchRaysIndex().x;
    const int iy = DispatchRaysIndex().y;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    float3 pixelColorPRD = float3(0.f);

    // normalized screen plane position, in [0,1]^2
    const float2 screen = float2(ix + .5f, iy + .5f) * float2(1.f / fbSize.x, 1.f / fbSize.y);

    RayDesc ray;
    ray.Origin = camera.position;
    ray.Direction = normalize(camera.direction + 
                              (screen.x - 0.5f) * camera.horizontal +
                              (screen.y - 0.5f) * camera.vertical);
    ray.TMin = 0.f;
    ray.TMax = 1e20f;

    TraceRay(
             traversable,   // This is the thing that stores our accel
             RAY_FLAG_NONE, // Ray Flags
             0xff,          // Instance inclusion mask
             0,             // Hit index?? Offset for hitgroup indexing?
             1,             // Ray type count
             0,             // Miss ray index
             ray,           // Ray
             pixelColorPRD  // Per pixel payload data that we want to move around
    );

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint rgba = 0xff000000
    | (r << 0) | (g << 8) | (b << 16);

    // and write frame buffer
    const uint fbIndex = ix + iy * fbSize.x;
    colorBuffer[fbIndex] = rgba;
}