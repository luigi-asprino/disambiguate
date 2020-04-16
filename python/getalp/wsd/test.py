from getalp.wsd.predicter_online import Predicter
import os
import glob
import time

predicter = Predicter("/tf/model", ["/tf/model/model_weights_wsd0"], True, 1, True, 1, False)

indir = "/tf/out/"
outdir = "/tf/out_inference"
os.mkdir(outdir)
files = glob.glob('/Users/lgu/Desktop/**/*.bz2', recursive=True)

t0 = time.time()
print(t0)
for f in files:
    print("Processing " + f)
    outsubdir = outdir + "/" + os.path.basename(os.path.dirname(f))
    os.mkdir(outsubdir)
    predicter.predictFile(f, outsubdir + "/" + os.path.dirname(f))
    print("End " + f)
t1 = time.time()
print(t1)
total = t1-t0
print(total)
