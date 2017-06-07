import os,sys
os.environ['GLOG_minloglevel'] = '2'

DATA_DIR= '/data/drinkingkazu/UBDeconvNet/dlmc_mcc8_multipvtx_v01'
if not sys.argv[1].find('caffemodel') >=0:
    sys.stderr.write('caffemodel file must be the 1st argument\n')
    sys.exit(1)

MODEL   = sys.argv[1]
GPUMEM=10000
STOP_COUNTER = -1
for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR = argv.replace('datadir=','')
    if argv.startswith('nevents='):
        STOP_COUNTER = int(argv.replace('nevents=',''))
    if argv.startswith('gpumem='):
        GPUMEM = int(argv.replace('gpumem=',''))

PROTO   = "%s/ana.prototxt" % DATA_DIR
CSVNAME = MODEL.replace('.caffemodel.h5','.csv')

if CSVNAME == MODEL:
    print 'Could not rename MODEL to CSV ...'
    sys.exit(1)
if os.path.isfile(CSVNAME):
    print 'Already processed:',CSVNAME
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
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import ROOT as rt
import time
from ROOT import larcv
import matplotlib.pyplot as plt
caffe.set_mode_gpu()

for argv in sys.argv:
    if argv.startswith('nevents='):
        STOP_COUNTER = int(argv.replace('nevents=',''))
    if argv.startswith('gpu='):
        caffe.set_device(int(argv.replace('gpu=','')))

BATCH_CTR=5
net = caffe.Net( PROTO, MODEL, caffe.TEST)
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
num_events = filler.get_n_entries()
if STOP_COUNTER <0:
    STOP_COUNTER = num_events

print
print 'Processing',MODEL
print 'Total number of events:',num_events
print 'Batch size:', BATCH_CTR
print

event_counter = 0
STOP_COUNTER  = 5320

current_index = 0
last_index = 0

npx_sum={1:0, 2:0}

filler.set_next_index(current_index)
fout=open(CSVNAME,'w')
fout.write('entry,type,score_shower,score_track\n')
for ibatch in range(0,num_events / BATCH_CTR+1):

    last_index = current_index    

    current_index = ibatch * BATCH_CTR

    #print 'current_index',current_index
    #filler.set_next_index(current_index)

    #print 'Batch:',ibatch

    num_entries = num_events - current_index

    if num_entries > BATCH_CTR: num_entries = BATCH_CTR

    net.forward()

    while filler.thread_running():
        time.sleep(0.001)

    events = filler.processed_events()
    entries = filler.processed_entries()

    if events.size() != BATCH_CTR:
        print "Batch counter mis-match!"
        raise Exception

    adcimgs = net.blobs["segdata"].data
    labels  = net.blobs["seglabel"].data
    scores  = net.blobs["softmax"].data

    for index in xrange(len(scores)):

        line ='%d' % entries[index]

        total_npx = 0
        total_correct_npx = 0

        label  = labels[index]  # Segmentation label raw image
        adcimg = adcimgs[index] # ADC raw image
        score  = scores[index]  # Score

        label = label[0]
        shower_score = score[1]
        track_score  = score[2]

        shower_score = shower_score[label>0]
        track_score  = track_score[label>0]
        label        = label[label>0]

        for px_index in xrange(len(label)):
            fout.write('%d,%d,%g,%g\n' % (event_counter,label[px_index],shower_score[px_index],track_score[px_index]))
            
        npx_sum[1] += (label==1).sum()
        npx_sum[2] += (label==2).sum()

        msg = 'Entry %4d ... Shower npx: %6d ... Track npx: %d ... Total npx: %6d \r' % (event_counter,npx_sum[1],npx_sum[2],npx_sum[1]+npx_sum[2])
        sys.stdout.write(msg)
        sys.stdout.flush()
        if (event_counter+1) % 100 == 0: 
            print

        event_counter += 1
        if event_counter >= STOP_COUNTER:
            break

    if num_entries < BATCH_CTR:
        break

    if event_counter >= STOP_COUNTER:
        break
fout.close()

