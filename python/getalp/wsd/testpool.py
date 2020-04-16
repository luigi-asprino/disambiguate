from threading import Thread
import time
import concurrent.futures


def predict(predicter, fin, outdir):
    outsubdir = outdir + "/" + os.path.basename(os.path.dirname(fin)) 
    if(not os.path.isdir(outsubdir)):
        os.mkdir(outsubdir)
        print("Out subdir created " + outsubdir)
                
    outfile = outsubdir + "/" + os.path.basename(fin)
    print("Processing " + f)
    print("Output " + outfile)
    predicter.predictFile(fin, outfile)
    print("End " + f)


if __name__ == "__main__":

    predicter = Predicter("/tf/model", ["/tf/model/model_weights_wsd0"], True, 1, True, 1, False)
    
    indir = "/tf/out/"
    outdir = "/tf/out_inference"
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
        print("Out dir created")
    
    files = glob.glob(indir + '/**/*.bz2', recursive=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    t0 = time.time()
    print(t0)
    for f in files:
        executor.submit(predict, predicter, f, outdir)
    t1 = time.time()
    print(t1)
    total = t1 - t0
    print(total)
    
