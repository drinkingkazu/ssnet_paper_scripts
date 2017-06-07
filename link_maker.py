import os

TARGET_DIR='/data/drinkingkazu/UBDeconvNet/suresnet64_plane2/spweights_yes'

files=[f for f in os.listdir(TARGET_DIR) if (f.endswith('500.caffemodel.h5') or f.endswith('000.caffemodel.h5')) and not f.startswith('classification')]
for f in files:
    if os.path.islink(f): continue
    os.system('ln -s %s/%s %s' % (TARGET_DIR,f,f))


