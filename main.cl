
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

#define land(buffer, pos, size) (buffer + (pos.y*size.x+pos.x))

#define water(buffer, pos, size) (buffer + size.x*size.y + (pos.y*size.x+pos.x))

#define sediment(buffer, pos, size) (buffer + size.x*size.y*2 + (pos.y*size.x+pos.x))


kernel void gravity(global const FLOAT* input, global FLOAT* output, const FLOAT repose, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    //const long id = coord.y * size.x + coord.x;

    *land(output, coord, size) = *land(input, coord, size);

    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            //const long h_id = h.y * size.x + h.x;
            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));

            FLOAT diff = (*land(input,h,size)-*land(input,coord,size))*dmod;
            diff = sign(diff) * max((FLOAT)0, fabs(diff)-repose);
            *land(output,coord,size) += amount * diff;
        }
    }
}

kernel void orogeny(global const FLOAT* input, global FLOAT* output, global const FLOAT* mask, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    *land(output,coord,size) = *land(input,coord,size) + *land(mask,coord,size)*amount;
}

kernel void hydro_precip(global const FLOAT* input, global FLOAT* output, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    
    *water(output,coord,size) = *water(input,coord,size) + amount;
}

kernel void hydro_sink(global const FLOAT* input, global FLOAT* output, const FLOAT max_depth){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;
    global const FLOAT* input_water = input + size.x*size.y;
    global FLOAT* output_water = output + size.x*size.y;

    output_water[id] = min(max_depth, input_water[id]);
    *water(output,coord,size) = min(max_depth, *water(input,coord,size));
}

constant FLOAT sediment_capacity_factor=1.0;
constant FLOAT erosion_speed_factor=0.05;

kernel void hydro_flow(global const FLOAT* input, global FLOAT* output, const FLOAT rate){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));

    const FLOAT alt = *land(input,coord,size) + *water(input,coord,size);

    FLOAT total_desired=0;
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            const FLOAT h_alt = *land(input,h,size) + *water(input,h,size);
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = rate * dmod * (alt-h_alt)/2;
            total_desired += desired_flow;
        }
    }

    const FLOAT flow_amount = min(total_desired, *water(input,coord,size));
    *water(output,coord,size) -= flow_amount;

    FLOAT accrued_sediment_change = 0;
    FLOAT accrued_land_change = 0;
    
    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){

            barrier(CLK_GLOBAL_MEM_FENCE);

            const long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }

            const FLOAT h_alt = *land(input,h,size) + *water(input,h,size);
            if(h_alt>=alt){
                continue;
            }

            const FLOAT dmod = (FLOAT)1 / length((FLOAT2)(dx,dy));
            const FLOAT desired_flow = rate * dmod * (alt-h_alt)/2;
            const FLOAT relegated_portion = desired_flow/total_desired;
            const FLOAT actual_flow = flow_amount * relegated_portion;

            const FLOAT portion_moving = actual_flow / *water(input,coord,size);
            const FLOAT sediment_moving = portion_moving * *sediment(input,coord,size);
            const FLOAT erosion = (actual_flow * sediment_capacity_factor - sediment_moving) * erosion_speed_factor;

            *water(output,h,size) += actual_flow;
            accrued_sediment_change += -sediment_moving;
            *sediment(output,h,size) += sediment_moving + erosion;
            accrued_land_change += -erosion/2;
            *land(output,h,size) += -erosion/2;
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    *sediment(output,coord,size) += accrued_sediment_change;
    *land(output,coord,size) += accrued_land_change;

    if(flow_amount < *water(input,coord,size)){
        const FLOAT remaining_water = *water(input,coord,size)-flow_amount;
        const FLOAT remaining_sediment = *sediment(input,coord,size) * remaining_water / *water(input,coord,size);
        const FLOAT deposition = remaining_sediment * erosion_speed_factor * rate;
        *sediment(output,coord,size) -= deposition;
        *land(output,coord,size) += deposition;
    }
}
