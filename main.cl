
#include "opensimplex.cl"
#include "convolve.cl"
#include "config.cl"

kernel void map_range(global FLOAT* field, const FLOAT src_low, const FLOAT src_high, 
const FLOAT dst_low, const FLOAT dst_high){
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = (field[lin_id]-src_low)/(src_high-src_low) * (dst_high-dst_low) + dst_low;
}

kernel void to_int32(global const FLOAT* field, global int* output){
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[lin_id] = (int)(field[lin_id] * (FLOAT)0xffff);
}

kernel void flatten_slopes(global const FLOAT* input, global FLOAT* output, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;

    FLOAT min=INFINITY;
    FLOAT max=-INFINITY;

    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            long h_id = h.y * size.x + h.x;
            FLOAT h_v = (input[h_id]-input[id])/length((FLOAT2)(dx,dy));
            if(h_v<min){
                min=h_v;
            }
            else if(h_v>max){
                max=h_v;
            }
        }
    }

    if(min==INFINITY || max==-INFINITY){
        output[id]=input[id];
        return;
    }

    output[id] = input[id] + amount * (min+max)/2.0;
}

kernel void gravity(global const FLOAT* input, global FLOAT* output, const FLOAT repose, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;

    output[id]=input[id];

    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;
            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));

            FLOAT diff = (input[h_id]-input[id])*dmod;
            diff = sign(diff) * max((FLOAT)0, fabs(diff)-repose);
            output[id] += amount * diff;
        }
    }
}

kernel void hydro_precip(global const FLOAT* input, global FLOAT* output, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;
    global const FLOAT* input_water = input + size.x*size.y;
    global FLOAT* output_water = output + size.x*size.y;

    output_water[id] = input_water[id] + amount;
}

kernel void hydro_sink(global const FLOAT* input, global FLOAT* output, const FLOAT max_depth){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;
    global const FLOAT* input_water = input + size.x*size.y;
    global FLOAT* output_water = output + size.x*size.y;

    output_water[id] = min(max_depth, input_water[id]);
}

//atomically modify a float value; uses legacy atomics so that they work on normal floats and don't require an extension
//doesn't compile when FLOAT isn't 'float'
void sync_float_add(volatile global FLOAT* p, FLOAT val){
    FLOAT ival,nval,rep;

    sfa_start:
    ival = *p;
    nval = ival + val;
    rep = atomic_xchg(p,nval);

    if(rep != ival){
        val = rep-ival;
        //recursion breaks compiler, so goto is used instead
        goto sfa_start;
    }
}

kernel void hydro_flow(global const FLOAT* input, global FLOAT* output){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;
    global const FLOAT* input_water = input + size.x*size.y;
    global FLOAT* output_water = output + size.x*size.y;

    const FLOAT alt = input[id] + input_water[id];

    FLOAT total_desired=0;
    //one of the downhill tiles is needed to calculate the final flow amount;
    // an arbitrary one is chosen as 'canary'
    FLOAT canary = NAN;
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;
            const FLOAT h_alt = input[h_id] + input_water[h_id];
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = dmod * (alt-h_alt)/2;
            canary = desired_flow;
            total_desired += desired_flow;
        }
    }

    FLOAT flow_amount;
    if(canary==NAN){
        return;//there are no downhill neighbors, no water will flow
    }
    else{
        const FLOAT canary_p = canary/total_desired;
        flow_amount = canary / (canary_p + 1);
        //can't flow what doesn't exist
        flow_amount = min(flow_amount, input_water[id]);
        sync_float_add(output_water+id, -flow_amount);
    }
    
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const long h_id = h.y * size.x + h.x;

            const FLOAT h_alt = input[h_id] + input_water[h_id];
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = dmod * (alt-h_alt)/2;
            const FLOAT relegated_portion = desired_flow/total_desired;
            const FLOAT actual_flow = flow_amount * relegated_portion;

            sync_float_add(output_water+h_id, actual_flow);
            
        }
    }
}
