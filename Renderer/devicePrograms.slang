struct Camera {
    float3 position;
    float3 direction;
    float3 horizontal;
    float3 vertical;
};

struct Payload {
    float3 attenuation;
    uint seed;

    float3 origin;
    float3 direction;
    float3 emitted;
    float3 radiance;
    bool done;
}

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
uint frameID;

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

static const int MIN = -2147483648;
static const int MAX = 2147483647;
static const float PI = 3.14159;
static const uint RAND_MAX = 4294967295;

inline int xorshift(in int value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return (value < 0) ? ~value + 1 : value;
}

inline int LCGRand(in int value) {
    value = (value * 1103515245 + 12345) & 0x7fffffff;
    return value;
}

int nextInt(inout int seed) {
    seed = LCGRand(seed);
    return seed;
}

float nextFloat(inout int seed) {
    seed = LCGRand(seed);
    // FIXME: This should have been a seed mapped from MIN to MAX to 0...1 instead
    return 2 * fract(float(seed) / (RAND_MAX - 1));
}

float nextFloat(inout int seed, in float max) {
    return nextFloat(seed) * max;
}

// float3 random_vec3(inout int seed) {
//     float x = rand(seed);
//     float y = rand(seed);
//     float z = rand(seed);
//     return float3(x, y, z);
// }

float3 unit_on_sphere(inout int seed) {
    float theta = 2 * PI * nextFloat(seed);
    float phi = acos(1 - 2 * nextFloat(seed));
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return float3(x, y, z);
}

inline float linear_to_gamma(in float linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

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
    // 
    // // ------------------------------------------------------------------
    // // compute normal, using either shading normal (if avail), or
    // // geometry normal (fallback)
    // // ------------------------------------------------------------------
    float3 sN, gN;
    uint numNormals, normalStride;
    normals.GetDimensions(numNormals, normalStride);
    // 
    const float3 A = vertices[index.x];
    const float3 B = vertices[index.y];
    const float3 C = vertices[index.z];
    gN = normalize(cross(B - A, C - A));
    // 

    // Shading normals being used
    if (numNormals > 0) {
        sN = (1.f - u - v) * normals[index.x]
        + u * normals[index.y]
            + v * normals[index.z];
    } else { // Geometric normals being used
        sN = gN;
    }

    // 
    // float3 diffuseColor = color;
    // uint numTexcoords, texcoordStride;
    // texcoords.GetDimensions(numTexcoords, texcoordStride);
    // if (hasTexture && numTexcoords > 0) {
    //     const float2 tc = (1.f - u - v) * texcoords[index.x]
    //                       + u * texcoords[index.y]
    //                       + v * texcoords[index.z];
    //     SamplerState temp;
    //     float4 fromTexture = texture.Sample(temp, tc);
    //     diffuseColor *= fromTexture.rgb;
    // }
    // 
    float3 rayDir = WorldRayDirection();
    float cosDN = (dot(sN, rayDir));
    bool front_face = cosDN <= 0.f;
    // float err = 1e-5f;
    // prd.emitted = float3(0.f, 0.f, 0.f);
    // prd.radiance += prd.emitted * prd.attenuation;
    // prd.attenuation *= cosDN * diffuseColor;
    // if (dot(sN, rayDir) > 0.f)
    //     prd.direction = -sN + unit_on_sphere(prd.seed);
    // else
    //     prd.direction = sN + unit_on_sphere(prd.seed);
    // prd.origin = WorldRayOrigin() + (RayTCurrent() - err) * rayDir;

    prd = 0.5f * float3(sN.x + 1.f, sN.y + 1.f, sN.z + 1.f);
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
    // prd = float3(1.f);

    // Set to a sky type blue
    float3 rayDir = WorldRayDirection();
    float a = 0.5f * (rayDir.y + 1.0);
    prd = (1.0f - a) * float3(1.f, 1.0f, 1.0f) + a * float3(0.5f, 0.7f, 1.0f);
    // prd.emitted = float3(1.f, 1.f, 1.f);
    // prd.radiance += prd.attenuation * prd.emitted;
    // prd.done = true;

    // Set to a complete black
    // prd = float3(0.f);
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
    // float3 pixelColorPRD;
    // float3 accumColor = float3(0.f);

    int seed = ix + iy * fbSize.x + frameID + 1; //+ frameID; // + frameID * fbSize.x * fbSize.y;

    // int raysPerPixel = 100;
    // int max_depth = 10;
    // Payload test_PRD;
    // test_PRD.seed = seed;
    // 
    // for (int i = 0; i < raysPerPixel; i++) {
    //     // Moved pixels
    //     float x_shift = nextFloat(test_PRD.seed);
    //     float y_shift = nextFloat(test_PRD.seed);
    // 
    //     // normalized screen plane position, in [0,1]^2
    //     const float2 screen = float2(ix + x_shift, iy + y_shift) * float2(1.f / fbSize.x, 1.f / fbSize.y);
    // 
    //     RayDesc ray;
    //     ray.Origin = camera.position;
    //     ray.Direction = normalize(camera.direction + 
    //                           (screen.x - 0.5f) * camera.horizontal +
    //                           (screen.y - 0.5f) * camera.vertical);
    //     ray.TMin = 0.f;
    //     ray.TMax = 1e20f;
    // 
    //     test_PRD.attenuation = float3(1.f, 1.f, 1.f);
    //     test_PRD.direction = ray.Direction;
    //     test_PRD.origin = ray.Origin;
    //     test_PRD.emitted = float3(0.f, 0.f, 0.f);
    //     test_PRD.done = false;
    //     test_PRD.radiance = float3(0.f, 0.f, 0.f);
    // 
    //     pixelColorPRD = float3(0.f, 0.f, 0.f);
    // 
    //     for (int j = 0; j < max_depth && test_PRD.done == false; j++) {
    //         // pixelColorPRD = float3(0.f);
    // 
    //         ray.Origin = test_PRD.origin;
    //         ray.Direction = test_PRD.direction;
    //         ray.TMin = 0.f;
    //         ray.TMax = 1e20f;
    // 
    //         TraceRay(
    //                  traversable,   // This is the thing that stores our accel
    //                  RAY_FLAG_NONE, // Ray Flags
    //                  0xff,          // Instance inclusion mask
    //                  0,             // Hit index?? Offset for hitgroup indexing?
    //                  1,             // Ray type count
    //                  0,             // Miss ray index
    //                  ray,           // Ray
    //                  test_PRD  // Per pixel payload data that we want to move around
    //         );
    //     }
    //     accumColor += (test_PRD.radiance) / raysPerPixel;
    // }

    float3 pixelColorPRD;
    // seed = LCGRand(seed);
    // float xshift = 2 * fract(float(seed) / (RAND_MAX - 1));

    float zshift = float(MIN);

    if (ix == 0.f && iy == 0.f)
        printf("%f\n", zshift);

    const float2 screen = float2(ix + 0.5f, iy + 0.5f) * float2(1.f / fbSize.x, 1.f / fbSize.y);

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