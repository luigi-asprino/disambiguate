from getalp.wsd.predicter_online import Predicter
import os
import glob
import time

predicter = Predicter("/tf/model", ["/tf/model/model_weights_wsd0"], True, 1, True, 1, False)

indir = "/tf/out/"
outdir = "/tf/out_inference"
if(not os.path.isdir(outdir)):
    os.mkdir(outdir)
    print("Out dir created")

files = glob.glob(indir + '/**/*.bz2', recursive=True)

t0 = time.time()
print(t0)
for f in files:
    outsubdir = outdir + "/" + os.path.basename(os.path.dirname(f)) 
    if(not os.path.isdir(outsubdir)):
        os.mkdir(outsubdir)
        print("Out subdir created " + outsubdir)
        
    outfile = outsubdir + "/" + os.path.basename(f)
    print("Processing " + f)
    print("Output " + outfile)
    predicter.predictFile(f, outfile)
    print("End " + f)
t1 = time.time()
print(t1)
total = t1 - t0
print(total)
