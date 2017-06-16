import os, sys

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

CSVNAME = MODEL.replace('.caffemodel.h5','_%s_plane%d.csv' % (FLAVOUR,PLANE))
if PLANE >2 or PLANE<0:
    print 'plane=%s' % str(PLANE),'not supported!'
    sys.exit(1)
if not FLAVOUR in AVAILABLE_FLAVOUR.keys():
    print 'flavour=%s' % str(FLAVOUR),'not supported!'
    sys.exit(1)
if CSVNAME == MODEL:
    print 'Could not rename MODEL to CSV ...'
    sys.exit(1)
if os.path.isfile(CSVNAME):
    print 'Already processed:',CSVNAME
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
#except Exception:
#    print 'Failed to import %s_template.py' % PROTO
#    sys.exit(1)

from choose_gpu import pick_gpu
GPUID=pick_gpu(GPUMEM,caffe_gpuid=True)
if GPUID < 0:
    sys.stderr.write('No available GPU with memory %d\n' % GPUMEM)
    sys.exit(1)

import matplotlib 
matplotlib.use('Agg')
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import ROOT as rt
import time
from ROOT import larcv
import matplotlib.pyplot as plt
import caffe
caffe.set_mode_gpu()
caffe.set_device(GPUID)

BATCH_CTR=5
net = caffe.Net(proto_cfg.name, MODEL, caffe.TEST)
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
num_events = filler.get_n_entries()
if STOP_COUNTER <0:
    STOP_COUNTER = num_events

print
print 'Processing',MODEL
print 'Total number of events:',num_events
print 'Stop after processing:',STOP_COUNTER
print 'Batch size:', BATCH_CTR
print

event_counter = 0
current_index = 0
last_index = 0

npx_sum={1:0, 2:0}
npx_sum_correct={1:0, 2:0}

filler.set_next_index(current_index)
roi_chain = None

fout=open(CSVNAME,'w')
fout.write('entry,label_shower_npx,reco_shower_npx,correct_shower_npx,label_track_npx,reco_track_npx,correct_track_npx,total_npx,correct_npx')
if '1e1p' in FLAVOUR:
    fout.write(',eminus_energy,eminus_mom,eminus_dirx,eminus_diry,eminus_dirz,proton_energy,proton_mom,proton_dirx,proton_diry,proton_dirz,open_angle')
    from ROOT import TChain
    roi_chain=TChain("partroi_segment_tree")
    for fname in filler.pd().io().file_list():
        roi_chain.AddFile(fname)    
fout.write('\n')

def target_event(roi_v):
    primary_eminus = None
    primary_proton = None
    for roi_index in xrange(roi_v.size()):
        roi = roi_v[roi_index]
        if roi.PdgCode() == 11 and roi.TrackID() == roi.ParentTrackID():
            if not primary_eminus is None: return False
            primary_eminus = roi_index
        elif roi.PdgCode() == 2212 and roi.TrackID() == roi.ParentTrackID():
            if not primary_eminus is None: return False
            primary_proton = roi_index
    if not primary_eminus is None and not primary_proton is None:
        return (primary_eminus,primary_proton)
    return False
    
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

        adcimg = adcimgs[index] # ADC raw image
        label  = labels[index]  # Segmentation label raw image
        score  = scores[index].argmax(axis=0) # Resulting type image
        score  = score.reshape([ adcimg.shape[0] ] + list(score.shape) ) # reshape

        ana_pixel_map = (adcimg > 20) * (label > 0.5)

        for t in [1,2]:
            #print '...',
            type_pixel_map = (label == t) * ana_pixel_map
            type_npx = type_pixel_map.sum()
            type_npx_correct = (label[type_pixel_map] == score[type_pixel_map]).sum()
            type_npx_res = (score[ana_pixel_map] == t).sum()

            line +=',%d.,%d.,%d.' % (type_npx,type_npx_res,type_npx_correct)
            #if type_npx:
            #    print int(float(type_npx_res)/float(type_npx) * 100.) / 100.,
            #else:
            #    print '  0.00',
            #if type_npx:
            #    print int(float(type_npx_correct)/float(type_npx) * 100.) / 100.,
            #else:
            #    print '  0.00',

            total_npx += type_npx
            total_correct_npx += type_npx_correct

            npx_sum[t] += type_npx
            npx_sum_correct[t] += type_npx_correct

        #print '...', int(float(total_correct_npx)/float(total_npx) * 100.)/100.
        line += ',%d.,%d.' % (total_npx,total_correct_npx)

        msg = 'Entry %4d ... Total npx: %6d ... Shower: %6d / %6d (%.2f%%) ... Track: %6d / %6d (%.2f%%)\r'
        temp_frac1 = 0.
        if npx_sum[1]: temp_frac1 = int(float(npx_sum_correct[1]) / npx_sum[1] * 10000.) / 100.
        temp_frac2 = 0.
        if npx_sum[2]: temp_frac2 = int(float(npx_sum_correct[2]) / npx_sum[2] * 10000.) / 100.
        msg = msg % (event_counter,
                     npx_sum[1] + npx_sum[2],
                     npx_sum_correct[1], npx_sum[1], temp_frac1,
                     npx_sum_correct[2], npx_sum[2], temp_frac2)
        sys.stdout.write(msg)
        sys.stdout.flush()

        if roi_chain:
            roi_chain.GetEntry(current_index + index)
            roi_v = roi_chain.partroi_segment_branch.ROIArray()
            #print events[index].event_key()
            #print roi_chain.partroi_segment_branch.event_key()
            #print roi_v.size()
            #for roi in roi_v:
            #    print roi.dump()
            target_info = target_event(roi_v)
            (eminus_energy,proton_energy) = (-1,-1)
            (eminus_mom,proton_mom) = (-1,-1)
            num_particles=-1
            eminus_dcos=[0,0,0]
            proton_dcos=[0,0,0]
            opening_ang=-1
            if target_info:
                eminus = roi_v[target_info[0]]
                proton = roi_v[target_info[1]]
                eminus_energy = eminus.EnergyInit() - 0.511
                proton_energy = proton.EnergyInit() - 938.28
                eminus_mom = np.sqrt(np.power(eminus.EnergyInit(),2) - np.power(0.511,2))
                proton_mom = np.sqrt(np.power(proton.EnergyInit(),2) - np.power(938.28,2))
                num_particles = roi_v.size()
                eminus_dcos = (eminus.Px() / eminus_mom, eminus.Py() / eminus_mom, eminus.Pz() / eminus_mom)
                proton_dcos = (proton.Px() / proton_mom, proton.Py() / proton_mom, proton.Pz() / proton_mom)
                # sanity check
                #print 1. - np.sqrt(np.power(eminus_dcos[0],2)+np.power(eminus_dcos[1],2)+np.power(eminus_dcos[2],2))
                #print 1. - np.sqrt(np.power(proton_dcos[0],2)+np.power(proton_dcos[1],2)+np.power(proton_dcos[2],2))
                opening_ang = np.arccos(eminus_dcos[0] * proton_dcos[0] + eminus_dcos[1] * proton_dcos[1] + eminus_dcos[2] * proton_dcos[2]) / np.pi * 180.
                #eminus_dir = eminus.
                
            line += ',%g,%g,%g,%g,%g' % (eminus_energy,eminus_mom,eminus_dcos[0],eminus_dcos[1],eminus_dcos[2])
            line += ',%g,%g,%g,%g,%g'  % (proton_energy,proton_mom,proton_dcos[0],proton_dcos[1],proton_dcos[2])
            line += ',%g' % opening_ang
        line+='\n'
        fout.write(line)

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

