import moa.clusterers.outliers.MCOD.MCOD;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

public class ThreadExecutor {

    EntryPoint ep;
    MCOD tMCOD;
    String filename;
    int numInstances;
    int numThreads;
    private final AtomicBoolean stop = new AtomicBoolean(false);
    private int count = 0;

    public ThreadExecutor(EntryPoint ep, MCOD tMCOD, String filename, int numInstances, int numThreads){
        this.ep = ep;
        this.tMCOD = tMCOD;
        this.filename = filename;
        this.numInstances = numInstances;
        this.numThreads = numThreads;
    }

    public Result compute(MCOD mcod) throws InterruptedException {
        this.ep.createTrainedClusterer(filename, tMCOD.kOption.getValue(), tMCOD.radiusOption.getValue(), tMCOD.windowSizeOption.getValue(), numInstances, numThreads, stop);
        count++;
        System.out.println("Created clusterer: " + count);
        //Thread.sleep(wait);
        return new Result(0);
    }

    public void execute() throws InterruptedException,
            ExecutionException {
        List<MCOD> mcods = new ArrayList<MCOD>();
        for (int i = 0; i < numInstances; i++) {
            mcods.add(new MCOD());
        }

        List<Callable<Result>> tasks = new ArrayList<Callable<Result>>();
        for (final MCOD mcod : mcods) {
            Callable<Result> c = new Callable<Result>() {
                @Override
                public Result call() throws Exception {
                    return compute(mcod);
                }
            };
            tasks.add(c);
        }

        ExecutorService exec = Executors.newFixedThreadPool(numThreads);

        try {
            long start = System.currentTimeMillis();
            List<Future<Result>> results = exec.invokeAll(tasks);
            System.out.println(String.format("map should be populated with clusters now"));
        } finally {
            exec.shutdown();
        }
    }

    private class Result {
        private final int wait;
        public Result(int code) {
            this.wait = code;
        }
    }
}
