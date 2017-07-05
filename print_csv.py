import sys,os
import pandas as pd

files = [f for f in os.listdir('.') if f.startswith(sys.argv[1]) and f.endswith('.csv')]
files_m = {}
for f in files:
    words=f.replace('.csv','').split('_')
    for i,word in enumerate(words):
        if word == 'iter' and words[i+1].isdigit():
            files_m[int(words[i+1])]=f
            break

keys=files_m.keys()
keys.sort()

for key in keys:
    df=pd.read_csv(files_m[key]).query('total_npx>0')
    print key
    print 'total',df.total_npx.values.sum(),'npx...',(df.correct_npx.values / df.total_npx.values).mean()
    subdf = df.query('label_shower_npx>0')
    print 'shower', subdf.label_shower_npx.values.sum(),'npx...',(subdf.correct_shower_npx.values / subdf.label_shower_npx.values).mean()
    subdf = df.query('label_track_npx>0')
    print 'track', subdf.label_track_npx.values.sum(),'npx...',(subdf.correct_track_npx.values / subdf.label_track_npx.values).mean()
    print

