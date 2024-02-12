#include "opensimplex.cl"
#include "valuenoise.cl"

constant ulong MAX_OCTAVES=16;
constant ulong octave_seed_mods[16]={
    0x0000000000000000,
    0xa9fbaed4049ce491,
    0x4439cf28cfe9acc6,
    0x3d27f93ab88fb739,
    0xffdd37b104d79c00,
    0x18476b03a9c46721,
    0x2acd9bc6e761cafd,
    0x2c623b4479cfff3b,
    0xf3a0bf5d1b9e9abb,
    0xf499e898432b95fc,
    0xf9f4db4d5bc2490b,
    0xe47342498feb63a0,
    0xed2bd0fe2def3c94,
    0xfdd27e1bdf9e4e09,
    0xe90c56de7649b910,
    0x6adeb3841d189c13
};

constant ulong3 warp_seed_mods = (ulong3)(
    0xa6073ca436d6f4c4,
    0xc1f9e4a3395ad25c,
    0xda975834c0abf855
);

FLOAT warp_noise3(const ulong seed, const FLOAT3 X, const FLOAT strength){
    FLOAT3 w = (FLOAT3)(
        NOISE_FUNCTION(seed^warp_seed_mods.x, X),//use extra padding, since the warp could cause more overlap
        NOISE_FUNCTION(seed^warp_seed_mods.y, X),
        NOISE_FUNCTION(seed^warp_seed_mods.z, X)
    ) * strength;
    return NOISE_FUNCTION(seed,X+w);
}

FLOAT fractal_noise3(const ulong seed, const FLOAT3 X, const FLOAT freq) {
    const ulong octaves = clamp((ulong)log2(MAX_NOISE_FREQ/freq),1UL,MAX_OCTAVES);
    FLOAT n=0;
    FLOAT nmax=0;
    for(int o=0; o<octaves; o++){
        const FLOAT f = freq * (1<<o);
        n += NOISE_FUNCTION(seed^octave_seed_mods[o],X*f)/(1<<o);
        nmax += 1.0/(1<<o);
    }
    return n/nmax;
}

FLOAT fractal_warp_noise3(const ulong seed, const FLOAT3 X, const FLOAT freq, const FLOAT warp_strength) {
    const ulong octaves = clamp((ulong)log2(MAX_NOISE_FREQ/freq),1UL,MAX_OCTAVES);
    FLOAT n=0;
    FLOAT nmax=0;
    for(int o=0; o<octaves; o++){
        const FLOAT f = freq * (1<<o);
        n += warp_noise3(seed^octave_seed_mods[o],X*f,warp_strength)/(1<<o);
        nmax += 1.0/(1<<o);
    }
    return n/nmax;
}

kernel void noisefill(global FLOAT* field, const FLOAT z, const ulong seed, const FLOAT freq){
    FLOAT3 pos = (FLOAT3)(get_global_id(0),get_global_id(1),z);
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = NOISE_FUNCTION(seed,pos*freq);
}

kernel void warp_noisefill(global FLOAT* field, const FLOAT z, const ulong seed, const FLOAT freq, const FLOAT warp_strength){
    FLOAT3 pos = (FLOAT3)(get_global_id(0),get_global_id(1),z);
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = warp_noise3(seed, pos*freq, warp_strength);
}

kernel void fractal_noisefill(global FLOAT* field, const FLOAT z, const ulong seed, const FLOAT freq){
    FLOAT3 pos = (FLOAT3)(get_global_id(0),get_global_id(1),z);
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = fractal_noise3(seed,pos,freq);
}

kernel void fractal_warp_noisefill(global FLOAT* field, const FLOAT z, const ulong seed, const FLOAT freq, const FLOAT warp_strength){
    FLOAT3 pos = (FLOAT3)(get_global_id(0),get_global_id(1),z);
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = fractal_warp_noise3(seed,pos,freq,warp_strength);
}


