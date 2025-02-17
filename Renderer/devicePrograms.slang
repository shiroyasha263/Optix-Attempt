struct Camera {
    float3 position;
    float3 direction;
    float3 horizontal;
    float3 vertical;
};

enum MaterialType {
    DIFFUSE,
    SPECULAR,
    DIELECTRIC
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
RWStructuredBuffer<float3> accumBuffer;
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
static const uint ALMOST_MAX = 0x0fffffff;

inline int xorshift(in uint value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return (value < 0) ? ~value + 1 : value;
}

inline int LCGRand(in uint value) {
    value = (value * 1103515245 + 12345) & 0x7fffffff;
    return value;
}

int nextInt(inout uint seed) {
    seed = xorshift(seed);
    return seed;
}

float nextFloat(inout uint seed) {
    uint x = nextInt(seed);
    return fract(float(x) / float(ALMOST_MAX));
}

float nextFloat(inout uint seed, in float max) {
    return nextFloat(seed) * max;
}

// float3 random_vec3(inout int seed) {
//     float x = rand(seed);
//     float y = rand(seed);
//     float z = rand(seed);
//     return float3(x, y, z);
// }

float3 unit_on_sphere(inout uint seed) {
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

    return 0.f;
}

float3 diffuseScatter(in float3 normal, in bool front_face, inout uint seed) {
    float3 direction;

    if (!front_face)
        direction = -normal + WorldRayDirection();
    else
        direction = normal + unit_on_sphere(seed);

    return direction;
}

float3 metalScatter(in float3 rayDir, in float3 normal, in float fuzz, in bool front_face, inout uint seed) {
    float3 direction;

    float scale = abs(dot(normal, rayDir));

    if (!front_face)
        direction = rayDir - 2 * scale * normal + fuzz * unit_on_sphere(seed);
    else
        direction = rayDir + 2 * scale * normal + fuzz * unit_on_sphere(seed);

    return direction;
}

float3 dielectricScatter(in float3 rayDir, in float3 normal, in float ior, in bool front_face, inout uint seed) {
    float3 direction;

    float ri = front_face ? (1.f / ior) : ior;

    float cos_theta = min(dot(-rayDir, normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;

    if (cannot_refract)
        return metalScatter(rayDir, normal, 0.f, front_face, seed);

    float3 r_out_perp = ri * (rayDir + cos_theta * normal);
    float dist = dot(r_out_perp, r_out_perp);
    float3 r_out_parallel = -sqrt(fabs(1.0f - dist)) * normal;

    return r_out_parallel + r_out_perp;
}

[shader("closesthit")]
void closesthit_radiance(
    inout Payload prd: SV_RayPayload, 
    in float2 barycentrics,
    uniform float3 color,
    uniform float3 emmissive,
    uniform float3 specular,
    uniform float shininess,
    uniform float ior,
    uniform MaterialType matType,
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
    float cosDN = (dot(sN, rayDir));
    bool front_face = cosDN <= 0.f;
    cosDN = abs(cosDN);

    float err = 1e-5f;
    prd.emitted = 10.f * emmissive;
    prd.radiance += prd.emitted * prd.attenuation;
    if (matType == MaterialType.DIFFUSE) {
        prd.attenuation *= cosDN * diffuseColor;
        prd.direction = diffuseScatter(sN, front_face, prd.seed);
        prd.origin = WorldRayOrigin() + (RayTCurrent()) * rayDir + err * prd.direction;
    }
    else if (matType == MaterialType.SPECULAR) {
        prd.attenuation *= cosDN * specular;
        prd.direction = metalScatter(rayDir, sN, shininess, front_face, prd.seed);
        if (dot(prd.direction, sN) < 0)
            prd.done = true;
        prd.origin = WorldRayOrigin() + (RayTCurrent()) * rayDir + err * prd.direction;
    }
    else if (matType == MaterialType.DIELECTRIC) {
        prd.direction = dielectricScatter(normalize(rayDir), normalize(sN), ior, front_face, prd.seed);
        prd.origin = WorldRayOrigin() + (RayTCurrent()) * rayDir + err * prd.direction;
    }
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
void miss_radiance(inout Payload prd: SV_RayPayLoad) {
    // set to constant white as background color
    // prd = float3(1.f);

    // Set to a sky type blue
    float3 rayDir = WorldRayDirection();
    float a = 0.5f * (rayDir.y + 1.0);
    // prd.radiance = (1.0f - a) * float3(1.f, 1.0f, 1.0f) + a * float3(0.5f, 0.7f, 1.0f);
    prd.emitted = (1.0f - a) * float3(1.f, 1.0f, 1.0f) + a * float3(0.5f, 0.7f, 1.0f);
    // prd.emitted = float3(0.f, 0.f, 0.f);
    prd.radiance += prd.attenuation * prd.emitted;
    prd.done = true;

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
    float3 pixelColorPRD;

    uint seed = ix + iy * fbSize.x + 1 + frameID; //+ frameID; // + frameID * fbSize.x * fbSize.y;
    seed = nextInt(seed) + frameID;
    const uint fbIndex = ix + iy * fbSize.x;

    float3 accumColor = (accumBuffer[fbIndex] * frameID) / (frameID + 1.f);

    int raysPerPixel = 4;
    int max_depth = 10;
    Payload test_PRD;
    test_PRD.seed = seed;

    for (int i = 0; i < raysPerPixel; i++) {
        // Moved pixels
        float x_shift = nextFloat(test_PRD.seed);
        float y_shift = nextFloat(test_PRD.seed);

        // normalized screen plane position, in [0,1]^2
        const float2 screen = float2(ix + x_shift, iy + y_shift) * float2(1.f / fbSize.x, 1.f / fbSize.y);

        RayDesc ray;
        ray.Origin = camera.position;
        ray.Direction = normalize(camera.direction + 
                              (screen.x - 0.5f) * camera.horizontal +
                                      (screen.y - 0.5f) * camera.vertical);
        ray.TMin = 0.f;
        ray.TMax = 1e20f;

        test_PRD.attenuation = float3(1.f, 1.f, 1.f);
        test_PRD.direction = ray.Direction;
        test_PRD.origin = ray.Origin;
        test_PRD.emitted = float3(0.f, 0.f, 0.f);
        test_PRD.done = false;
        test_PRD.radiance = float3(0.f, 0.f, 0.f);

        pixelColorPRD = float3(0.f, 0.f, 0.f);

        for (int j = 0; j < max_depth && test_PRD.done == false; j++) {
            // pixelColorPRD = float3(0.f);

            ray.Origin = test_PRD.origin;
            ray.Direction = test_PRD.direction;
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
                     test_PRD       // Per pixel payload data that we want to move around
            );
        }
        accumColor += ((test_PRD.radiance) / (raysPerPixel * (frameID + 1.f)));
    }

    accumBuffer[fbIndex] = accumColor;

    const int r = int(255.99f * clamp(linear_to_gamma(accumColor.x), 0.f, 1.f));
    const int g = int(255.99f * clamp(linear_to_gamma(accumColor.y), 0.f, 1.f));
    const int b = int(255.99f * clamp(linear_to_gamma(accumColor.z), 0.f, 1.f));

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint rgba = 0xff000000
    | (r << 0) | (g << 8) | (b << 16);

    // and write frame buffer
    colorBuffer[fbIndex] = rgba;
}