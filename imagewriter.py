import os,sys
os.environ['GLOG_minloglevel'] = '2'

DATA_DIR= os.getcwd()
if not sys.argv[1].find('caffemodel') >=0:
    sys.stderr.write('caffemodel file must be the 1st argument\n')
    sys.exit(1)

MODEL   = sys.argv[1]
GPUMEM=10000
STOP_COUNTER = 100
for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR = argv.replace('datadir=','')
    if argv.startswith('nevents='):
        STOP_COUNTER = int(argv.replace('nevents=',''))
    if argv.startswith('gpumem='):
        GPUMEM = int(argv.replace('gpumem=',''))

PROTO   = "%s/ana.prototxt" % DATA_DIR
ROOTNAME = MODEL.replace('.caffemodel.h5','.root')

if ROOTNAME == MODEL:
    print 'Could not rename MODEL to ROOT ...'
    sys.exit(1)
if os.path.isfile(ROOTNAME):
    print 'Already processed:',ROOTNAME
    sys.exit(1)
if not os.path.isfile(PROTO):
    print 'PROTO not found:',PROTO
    sys.exit(1)

from choose_gpu import pick_gpu
GPUID=pick_gpu(GPUMEM,caffe_gpuid=True)
if GPUID < 0:
    sys.stderr.write('No available GPU with memory %d\n' % GPUMEM)
    sys.exit(1)

import matplotlib 
matplotlib.use('Agg')
import caffe
import numpy as np
import ROOT as rt
import lmdb
import time
from ROOT import larcv
import matplotlib.pyplot as plt
caffe.set_mode_gpu()
caffe.set_device(GPUID)

PROTO = "ana.prototxt"
MODEL = None
ANA_OUTPUT_CFG="ana_out.cfg"
debug = False

for argv in sys.argv:

    if argv == 'debug': debug=True
    if argv.find('.caffemodel')>=0: MODEL=argv

proc = larcv.ProcessDriver('OutputProcessDriver')
proc.configure(ANA_OUTPUT_CFG)
proc.override_output_file(ROOTNAME)
proc.initialize()

py_image_maker = proc.process_ptr(proc.process_id("PyImageMaker"))
outman = larcv.IOManager(larcv.IOManager.kWRITE)

net = caffe.Net( PROTO, MODEL, caffe.TEST)
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
num_events = filler.get_n_entries()

print
print 'Total number of events:',num_events
print

event_counter = 0
BATCH_CTR = None
current_index = 0

filler.set_next_index(current_index)
while event_counter < num_events:

    if (event_counter+1) % 100 == 0:
        print "Current index @", event_counter

    net.forward()
    while filler.thread_running():
        time.sleep(0.001)

    meta_vv = filler.meta()
    adcimg  = net.blobs["segdata"].data
    labels  = net.blobs["seglabel"].data
    softmax = net.blobs["softmax"].data

    events = filler.processed_events()
    entries = filler.processed_entries()

    if BATCH_CTR is None: 
        BATCH_CTR = int(events.size())

    num_entries = num_events - event_counter

    if BATCH_CTR and num_entries > BATCH_CTR: 
        num_entries = BATCH_CTR

    if events.size() != BATCH_CTR:
        print "Batch counter mis-match!"
        raise Exception

    for idx in xrange(num_entries):
        
        if debug:
            adcpng = plt.imshow(adcimg[idx][0])
            adcpng.write_png('entry%06d_adc.png' % (ibatch * BATCH_CTR + idx))

        meta_v = meta_vv.at(idx)
        img_array = softmax[idx]
        for ch in xrange(len(img_array)):
            img = img_array[ch]
            py_image_maker.append_ndarray_meta(img.transpose(),meta_v[0])
            
            if debug:
                png=plt.imshow(img)
                png.write_png('entry%06d_ch%02d.png' % ((ibatch * BATCH_CTR + idx),ch))

        py_image_maker.append_ndarray_meta(adcimg[idx][0].transpose(),meta_v[0])
        py_image_maker.append_ndarray_meta(labels[idx][0].transpose(),meta_v[0])
        print 'run',events[idx].run(),'subrun',events[idx].subrun(),'event',events[idx].event()
        py_image_maker.set_id(events[idx].run(),events[idx].subrun(),events[idx].event())
        proc.process_entry()
        event_counter += 1

        if event_counter >= STOP_COUNTER:
            break

    if num_entries < BATCH_CTR:
        break

    if event_counter >= STOP_COUNTER:
        break

proc.finalize()

