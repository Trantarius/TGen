
#include "noise.cl"
#include "convolve.cl"
#include "config.cl"

struct Data{
    FLOAT* land;
    FLOAT* water;
    FLOAT* sediment;
};

global struct Data prev;
global struct Data next;
global long2 size;

// contains oro_levels+1 noisemaps. the map at 0 is the sum of the remaining maps.
// each map is an octave of the generated noise, seperately updated when necessary.
global FLOAT* global oro_buffer;
global FLOAT* global oro_update_times;
global long oro_levels;
global ulong oro_seed;

kernel void to_png(global const FLOAT* field, global int* output, const long sub_idx, const FLOAT vmin, const FLOAT vmax){
    const long id = get_global_linear_id() + sub_idx * size.x * size.y;
    const FLOAT normed = (field[id]-vmin)/(vmax-vmin);
    output[get_global_linear_id()] = (int)(normed * (FLOAT)0xffff);
}

kernel void supply_buffers(global FLOAT* p_buf, global FLOAT* n_buf, long width, long height){
    prev.land = p_buf;
    prev.water = p_buf + width*height;
    prev.sediment = p_buf + width*height*2;
    next.land = n_buf;
    next.water = n_buf + width*height;
    next.sediment = n_buf + width*height*2;
    size = (long2)(width,height);
}

#define STD_KERNEL_START \
const long id = get_global_linear_id(); \
const long2 coord = (long2)(get_global_id(0), get_global_id(1)); \
if(coord.x<0 || coord.y<0 || coord.x>=size.x || coord.y>=size.y){ return; }

kernel void copy_back(){
    prev.land[get_global_linear_id()]=next.land[get_global_linear_id()];
    prev.water[get_global_linear_id()]=next.water[get_global_linear_id()];
    prev.sediment[get_global_linear_id()]=next.sediment[get_global_linear_id()];
};

kernel void init_height(const ulong seed, const FLOAT freq, const FLOAT warp_strength, const FLOAT max_height){
    if(get_global_linear_id()==0){
        oro_seed=seed;
    }
    const FLOAT3 pos = (FLOAT3)(get_global_id(0), get_global_id(1), 0);
    prev.land[get_global_linear_id()] = fractal_warp_noise3(seed, pos, freq, warp_strength) * max_height;
    next.land[get_global_linear_id()] = prev.land[get_global_linear_id()];
}

kernel void orogeny(const FLOAT T, const FLOAT freq, const FLOAT warp_strength, const FLOAT amount){
    const FLOAT3 pos = (FLOAT3)(get_global_id(0), get_global_id(1), T);
    next.land[get_global_linear_id()] = prev.land[get_global_linear_id()] +
        fractal_warp_noise3(oro_seed, pos, freq, warp_strength) * amount;
}

kernel void gravity(const FLOAT repose, const FLOAT amount){

    STD_KERNEL_START;

    next.land[id] = prev.land[id];

    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;
            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));

            FLOAT diff = (prev.land[h_id] - prev.land[id]) * dmod;
            diff = sign(diff) * max((FLOAT)0, fabs(diff)-repose);
            next.land[id] += amount * diff;
        }
    }
}

kernel void hydro_precip(const FLOAT amount){
    STD_KERNEL_START
    next.water[id] = prev.water[id] + amount;
}

kernel void hydro_sink(const FLOAT max_depth){
    STD_KERNEL_START
    next.water[id] = min(max_depth, prev.water[id]);
}

kernel void hydro_flow_part(const char part, const FLOAT rate){

    const long2 coord = (long2)(get_global_id(0), get_global_id(1)) * 3 + (long2)(part%3,part/3);
    const long id = coord.y * size.x + coord.x;
    if(coord.x<0 || coord.y<0 || coord.x>=size.x || coord.y>=size.y){ return; }
    //printf("%i (%i, %i)",(int)part, coord.x, coord.y);

    const FLOAT alt = prev.land[id] + prev.water[id];

    FLOAT total_desired=0;
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;
            const FLOAT h_alt = prev.land[h_id] + prev.water[h_id];
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = rate * dmod * (alt-h_alt)/2;
            total_desired += desired_flow;
        }
    }
    const FLOAT flow_amount = min(total_desired, prev.water[id]);
    next.water[id] -= flow_amount;

    FLOAT accrued_sediment_change = 0;
    FLOAT accrued_land_change = 0;
    
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){

            if(flow_amount <= 0){
                continue;
            }

            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;
            const FLOAT h_alt = prev.land[h_id] + prev.water[h_id];
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = rate * dmod * (alt-h_alt)/2;
            const FLOAT relegated_portion = desired_flow/total_desired;
            const FLOAT actual_flow = flow_amount * relegated_portion;

            const FLOAT portion_moving = actual_flow / prev.water[id];
            const FLOAT sediment_moving = portion_moving * prev.sediment[id];
            const FLOAT erosion = (actual_flow * SEDIMENT_CAPACITY - sediment_moving) * EROSION_SPEED;

            next.water[h_id] += actual_flow;
            accrued_sediment_change += -sediment_moving;
            next.sediment[h_id] += sediment_moving + erosion;
            accrued_land_change += -erosion/2;
            next.land[h_id] += -erosion/2;
        }
    }

    next.sediment[id] += accrued_sediment_change;
    next.land[id] += accrued_land_change;

    if(flow_amount < prev.water[id]){
        const FLOAT remaining_water = prev.water[id] - flow_amount;
        const FLOAT remaining_sediment = prev.sediment[id] * remaining_water / prev.water[id];
        const FLOAT deposition = remaining_sediment * EROSION_SPEED * rate;
        next.sediment[id] -= deposition;
        next.land[id] += deposition;
    }
}