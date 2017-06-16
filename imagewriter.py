import os,sys
os.environ['GLOG_minloglevel'] = '2'

DATA_DIR= os.getcwd()
if not sys.argv[1].find('caffemodel') >=0:
    sys.stderr.write('caffemodel file must be the 1st argument\n')
    sys.exit(1)

AVAILABLE_FLAVOUR = {'val'  : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val.root',
                     '1e1p' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_1e1p.root',
                     'nue'  : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_nue.root',
                     'numu' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_numu.root',
                     '1e1p_lowE' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_1e1p_lowE.root'
                     }

MODEL        = sys.argv[1]
PROTO        = 'ures'
GPUMEM       = 10000
STOP_COUNTER = -1
PLANE        = 2
FLAVOUR      = None

for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR = argv.replace('datadir=','')
    if argv.startswith('nevents='):
        STOP_COUNTER = int(argv.replace('nevents=',''))
    if argv.startswith('gpumem='):
        GPUMEM = int(argv.replace('gpumem=',''))
    if argv.startswith('flavour='):
        FLAVOUR = argv.replace('flavour=','')
    if argv.startswith('plane='):
        PLANE = int(argv.replace('plane=',''))
    if argv.startswith('proto='):
        PROTO = argv.replace('proto=','')

ROOTNAME = MODEL.replace('.caffemodel.h5','_%s_plane%d.root' % (FLAVOUR,PLANE))
if PLANE >2 or PLANE<0:
    print 'plane=%s' % str(PLANE),'not supported!'
    sys.exit(1)
if not FLAVOUR in AVAILABLE_FLAVOUR.keys():
    print 'flavour=%s' % str(FLAVOUR),'not supported!'
    sys.exit(1)
if ROOTNAME == MODEL:
    print 'Could not rename MODEL to CSV ...'
    sys.exit(1)
if os.path.isfile(ROOTNAME):
    print 'Already processed:',ROOTNAME
    sys.exit(1)

#
# Generate temp file 
#
import tempfile, filler_template
filler_cfg=tempfile.NamedTemporaryFile()
filler_cfg.write(filler_template.get_template().replace('FILEPATH',AVAILABLE_FLAVOUR[FLAVOUR]).replace('PLANE',str(PLANE)))
filler_cfg.flush()

exec('from %s_template import get_template as proto_template' % PROTO)
proto_cfg=tempfile.NamedTemporaryFile()
proto_cfg.write(proto_template().replace('CONFIG_NAME',filler_cfg.name))
proto_cfg.flush()

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

net = caffe.Net(proto_cfg.name, MODEL, caffe.TEST)
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

