#include "config.cl"
/////////////// K.jpg's Re-oriented 8-Polong BCC Noise (OpenSimplex2S) ////////////////
////////////////////// Output: FLOAT4(dF/dx, dF/dy, dF/dz, value) //////////////////////

// Borrowed from Stefan Gustavson's noise code
FLOAT4 permute(FLOAT4 t) {
    return t * (t * (FLOAT)34.0 + (FLOAT)133.0);
}

// Gradient set is a normalized expanded rhombic dodecahedron
FLOAT3 grad(FLOAT hash) {
    
    // Random vertex of a cube, +/- 1 each
    FLOAT3 cube = fmod(floor(hash / (FLOAT3)(1.0, 2.0, 4.0)), 2.0) * (FLOAT)2.0 - (FLOAT)1.0;
    
    // Random edge of the three edges connected to that vertex
    // Also a cuboctahedral vertex
    // And corresponds to the face of its dual, the rhombic dodecahedron
    FLOAT3 cuboct = cube;
    cuboct[(long)(hash / 16.0)] = 0.0;
    
    // In a funky way, pick one of the four polongs on the rhombic face
    FLOAT type = fmod(floor(hash / 8.0), 2.0);
    FLOAT3 rhomb = ((FLOAT)1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));
    
    // Expand it so that the new edges are the same length
    // as the existing ones
    FLOAT3 grad = cuboct * (FLOAT)1.22474487139 + rhomb;
    
    // To make all gradients the same length, we only need to shorten the
    // second type of vector. We also put in the whole noise scale constant.
    // The compiler should reduce it longo the existing FLOATs. I think.
    grad *= ((FLOAT)1.0 - (FLOAT)0.042942436724648037 * type) * (FLOAT)3.5946317686139184;
    
    return grad;
}

// BCC lattice split up longo 2 cube lattices
FLOAT4 openSimplex2SDerivativesPart(FLOAT3 X) {
    FLOAT3 b = floor(X);
    FLOAT4 i4 = (FLOAT4)(X - b, 2.5);
    
    // Pick between each pair of oppposite corners in the cube.
    FLOAT3 v1 = b + floor(dot(i4, (FLOAT4)(.25)));
    FLOAT3 v2 = b + (FLOAT3)(1, 0, 0) + (FLOAT3)(-1, 1, 1) * floor(dot(i4, (FLOAT4)(-.25, .25, .25, .35)));
    FLOAT3 v3 = b + (FLOAT3)(0, 1, 0) + (FLOAT3)(1, -1, 1) * floor(dot(i4, (FLOAT4)(.25, -.25, .25, .35)));
    FLOAT3 v4 = b + (FLOAT3)(0, 0, 1) + (FLOAT3)(1, 1, -1) * floor(dot(i4, (FLOAT4)(.25, .25, -.25, .35)));
    
    // Gradient hashes for the four vertices in this half-lattice.
    FLOAT4 hashes = permute(fmod((FLOAT4)(v1.x, v2.x, v3.x, v4.x), 289.0));
    hashes = permute(fmod(hashes + (FLOAT4)(v1.y, v2.y, v3.y, v4.y), 289.0));
    hashes = fmod(permute(fmod(hashes + (FLOAT4)(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);
    
    // Gradient extrapolations & kernel function
    FLOAT3 d1 = X - v1; FLOAT3 d2 = X - v2; FLOAT3 d3 = X - v3; FLOAT3 d4 = X - v4;
    FLOAT4 a = max((FLOAT)0.75 - (FLOAT4)(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    FLOAT4 aa = a * a; FLOAT4 aaaa = aa * aa;
    FLOAT3 g1 = grad(hashes.x); FLOAT3 g2 = grad(hashes.y);
    FLOAT3 g3 = grad(hashes.z); FLOAT3 g4 = grad(hashes.w);
    FLOAT4 extrapolations = (FLOAT4)(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    
    FLOAT4 d1234[3]={
        (FLOAT4)(d1.x,d2.x,d3.x,d4.x),
        (FLOAT4)(d1.y,d2.y,d3.y,d4.y),
        (FLOAT4)(d1.z,d2.z,d3.z,d4.z)
    };

    FLOAT4 aa_a_ext = aa * a * extrapolations;

    FLOAT4 g1234[3]={
        (FLOAT4)(g1.x,g2.x,g3.x,g4.x),
        (FLOAT4)(g1.y,g2.y,g3.y,g4.y),
        (FLOAT4)(g1.z,g2.z,g3.z,g4.z)
    };

    // Derivatives of the noise
    FLOAT3 derivative = -(FLOAT)8.0 * (FLOAT3)(dot(d1234[0],aa_a_ext),dot(d1234[1],aa_a_ext),dot(d1234[2],aa_a_ext))
        + (FLOAT3)(dot(g1234[0],aaaa),dot(g1234[1],aaaa),dot(g1234[2],aaaa));
    
    // Return it all as a FLOAT4
    return (FLOAT4)(derivative, dot(aaaa, extrapolations));
}

// Use this if you don't want Z to look different from X and Y
FLOAT4 openSimplex2SDerivatives_Conventional(FLOAT3 X) {
    X = dot(X, (FLOAT3)(2.0/3.0)) - X;
    
    FLOAT4 result = openSimplex2SDerivativesPart(X) + openSimplex2SDerivativesPart(X + (FLOAT)144.5);
    
    return (FLOAT4)(dot(result.xyz, (FLOAT3)(2.0/3.0)) - result.xyz, result.w);
}

// Use this if you want to show X and Y in a plane, then use Z for time, vertical, etc.
FLOAT4 openSimplex2SDerivatives_ImproveXY(FLOAT3 X) {
    
    // Not a skew transform.
    FLOAT3 orthonormalMap[3] = {
        (FLOAT3)(0.788675134594813, -0.211324865405187, -0.577350269189626),
        (FLOAT3)(-0.211324865405187, 0.788675134594813, -0.577350269189626),
        (FLOAT3)(0.577350269189626, 0.577350269189626, 0.577350269189626)};
    
    FLOAT3 orthonormalMapT[3] = {
        (FLOAT3)(0.788675134594813, -0.211324865405187, 0.577350269189626),
        (FLOAT3)(-0.211324865405187, 0.788675134594813, 0.577350269189626),
        (FLOAT3)(-0.577350269189626, -0.577350269189626, 0.577350269189626)};
    
    X = (FLOAT3)(dot(orthonormalMap[0],X),dot(orthonormalMap[1],X),dot(orthonormalMap[2],X));
    FLOAT4 result = openSimplex2SDerivativesPart(X) + openSimplex2SDerivativesPart(X + (FLOAT)144.5);
    
    return (FLOAT4)(dot(result.xyz,orthonormalMapT[0]),dot(result.xyz,orthonormalMapT[1]),dot(result.xyz,orthonormalMapT[2]), result.w);
}

//////////////////////////////// End noise code ////////////////////////////////

// Z is used as a psuedo-seed value
FLOAT noise2(FLOAT2 X, FLOAT Z) {
    return openSimplex2SDerivatives_ImproveXY((FLOAT3)(X,Z)).w;
}

FLOAT warp_noise2(const FLOAT2 X, const FLOAT Z, const FLOAT strength){
    FLOAT3 w = (FLOAT3)(
        noise2(X,Z+2 + strength),//use extra padding, since the warp could cause more overlap
        noise2(X,Z+4 + strength),
        noise2(X,Z+6 + strength)
    ) * strength;
    return noise2(X+w.xy,Z+w.z);
}

FLOAT fractal_noise2(const FLOAT2 X, const FLOAT Z, const FLOAT min_f, const FLOAT max_f) {

    FLOAT n=0;
    FLOAT z=Z;
    long o=1;

    while(o * min_f <= max_f){
        const FLOAT f = min_f * o;
        n += noise2(X*f,z)/o;
        z += 2;//nodes are approx 1 unit apart, jump by 2 should be enough to be independent
        o *= 2;
    }

    return n/2;
}

FLOAT fractal_warp_noise2(const FLOAT2 X, const FLOAT Z, const FLOAT min_f, const FLOAT max_f, const FLOAT warp_strength) {

    FLOAT n=0;
    FLOAT z=Z;
    long o=1;

    while(o * min_f <= max_f){
        const FLOAT f = min_f * o;
        n += warp_noise2(X*f, z, warp_strength)/o;
        z += 8 + warp_strength;//needs extra padding for warp channels
        o *= 2;
    }

    return n/2;
}

kernel void noisefill(global FLOAT* field, const FLOAT Z, const FLOAT freq){
    FLOAT2 pos = (FLOAT2)(get_global_id(0),get_global_id(1));
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = noise2(pos*freq,Z);
}

kernel void warp_noisefill(global FLOAT* field, const FLOAT Z, const FLOAT freq, const FLOAT warp_strength){
    FLOAT2 pos = (FLOAT2)(get_global_id(0),get_global_id(1));
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = warp_noise2(pos*freq,Z,warp_strength);
}

kernel void fractal_noisefill(global FLOAT* field, const FLOAT Z, const FLOAT freq){
    FLOAT2 pos = (FLOAT2)(get_global_id(0),get_global_id(1));
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = fractal_noise2(pos,Z,freq,0.25);
}

kernel void fractal_warp_noisefill(global FLOAT* field, const FLOAT Z, const FLOAT freq, const FLOAT warp_strength){
    FLOAT2 pos = (FLOAT2)(get_global_id(0),get_global_id(1));
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = fractal_warp_noise2(pos,Z,freq,0.25, warp_strength);
}


