#!/home/tranus/pyenv/bin/python

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import pyopencl as cl
import multiprocess as mp
import random
from time import sleep
import PIL
from sys import getsizeof as sizeof
from math import *

#in pixels (AKA simulation units)
#affects the size of the area simulated, not the accuracy
IMG_SIZE=1024
#maximum, in sim units
PERIOD=1024
#modifier, unitless
SLOPE=1
SIM_STEPS=100000
#modifier, lower increases stability
SIM_RATE=0.25

#in sim units
VERT_RANGE= SLOPE*PERIOD/8
print("VERT_RANGE: {:.0f}".format(VERT_RANGE))
SEED=random.random()*(1<<10)

BUF_SHAPE=(3,IMG_SIZE,IMG_SIZE)
BUF_LAND=0
BUF_WATER=1
BUF_SEDIMENT=2

FLOAT='float'
NPFLOAT=np.float32

draw_queue = mp.Queue()
should_close=mp.Event()
stop_drawing=mp.Event()
draw_ready = mp.Event()

def draw():

    matplotlib.use('GTK3Agg')
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    plt.axes((0,0,1,1))
    def on_close(*_):
        global should_close
        should_close.set()
    fig.canvas.mpl_connect('close_event',on_close)
    plt.axis('off')

    water_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('water_cmap',[(0,0,1,0),(0,0,1,0.5)])

    land_img_ax = plt.imshow(np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT), cmap='gray', vmin=0, vmax=VERT_RANGE)
    water_img_ax = plt.imshow(np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT), cmap=water_cmap, vmin=0, vmax=1)

    while(not stop_drawing.is_set()):
        if(not draw_queue.empty()):
            img=draw_queue.get()
            land_img_ax.set_data(img[BUF_LAND])
            water_img_ax.set_data(img[BUF_WATER])
        else:
            draw_ready.set()
            fig.canvas.flush_events()
            fig.canvas.draw_idle()
            sleep(0.016)

draw_proc=mp.Process(target=draw)
draw_proc.daemon=True
draw_proc.start()

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

start=timer()
with open('main.cl','r') as maincl:
    prg = cl.Program(ctx, str(maincl.read())).build(options=[f'-DFLOAT={FLOAT} -DIMG_SIZE={IMG_SIZE}'])
krn={}
for kernel in prg.all_kernels():
    krn[kernel.function_name]=kernel
print('kernel compilation: {:0.4f}s'.format(timer()-start))

host_buf = np.zeros(BUF_SHAPE,dtype=NPFLOAT)
tmp_buf = np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT)
current_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
oro_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tmp_buf)
del tmp_buf
next_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
def swap_buffers():
    cl.enqueue_copy(queue, current_buf, next_buf)

start=timer()
krn['fractal_warp_noisefill'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, NPFLOAT(SEED), NPFLOAT(1.0/PERIOD), NPFLOAT(1))
cl.enqueue_copy(queue,host_buf,current_buf)
krn['map_range'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, 
    NPFLOAT(np.min(host_buf)), NPFLOAT(np.max(host_buf)), NPFLOAT(0), NPFLOAT(VERT_RANGE))
cl.enqueue_copy(queue, next_buf, current_buf)
print("noisegen {:0.4f}s".format(timer()-start))

total_step_time = 0
total_steps = 0
very_start=timer()
for step in range(SIM_STEPS):
    start=timer()
    if(should_close.is_set()):
        break

    if(step%1000==0):
        krn['fractal_warp_noisefill'](queue, (IMG_SIZE,IMG_SIZE), None, oro_buf, 
        NPFLOAT(SEED+1000*SIM_RATE*step/(1<<20)), NPFLOAT(1.0/PERIOD), NPFLOAT(1))

    krn['orogeny'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, oro_buf, NPFLOAT(0.001*SIM_RATE))
    swap_buffers()
    krn['gravity'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, NPFLOAT(1), NPFLOAT(0.1 * SIM_RATE))
    swap_buffers()
    krn['hydro_precip'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, NPFLOAT(0.001 * SIM_RATE))
    swap_buffers()
    krn['hydro_flow'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, NPFLOAT(SIM_RATE))
    swap_buffers()
    krn['hydro_sink'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, NPFLOAT(1))
    swap_buffers()

    
    queue.finish()
    elapsed=timer()-start
    total_step_time+=elapsed
    total_steps+=1
    print('\r',' '*50,end='')
    print('\rstep {}\t{:0.4f}s'.format(step,elapsed),end='')

    #send image to be drawn if the drawer is ready, or if on the last step
    if(draw_ready.is_set() or step==SIM_STEPS-1):
        cl.enqueue_copy(queue,host_buf,current_buf)
        draw_queue.put(host_buf)
        draw_ready.clear()

cl.enqueue_copy(queue, host_buf, current_buf, is_blocking=False)
queue.finish()

total_elapsed = timer()-very_start
print('\navg step time: {:0.4f}s'.format(total_step_time/total_steps))
print('total time: {:0.4f}s'.format(total_elapsed))
print('overhead: {:0.4f}s'.format(total_elapsed-total_step_time))

imdata = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.int32)
imbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=imdata)
krn['map_range'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, NPFLOAT(0), NPFLOAT(VERT_RANGE), NPFLOAT(0), NPFLOAT(1))
krn['to_int32'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, imbuf)
cl.enqueue_copy(queue, imdata, imbuf)

img = PIL.Image.fromarray(imdata,'I')
img.save('out.png')


while(not should_close.is_set()):
    sleep(0.1)

stop_drawing.set()
draw_proc.join()