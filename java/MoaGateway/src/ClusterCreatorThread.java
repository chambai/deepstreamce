import moa.clusterers.outliers.MCOD.MCOD;
import java.util.concurrent.atomic.AtomicBoolean;

public class ClusterCreatorThread extends Thread {

    EntryPoint ep;
    MCOD tMCOD;
    String filename;
    int numInstances;
    int threadNum;
    private final AtomicBoolean stop = new AtomicBoolean(false);

    public ClusterCreatorThread(EntryPoint ep, MCOD tMCOD, String filename, int numInstances, int threadNum){
        this.ep = ep;
        this.tMCOD = tMCOD;
        this.filename = filename;
        this.numInstances = numInstances;
        this.threadNum = threadNum;
    }

    public void run(){
        try{
            System.out.println("Thread " + filename + "_" + threadNum + " running");
            this.ep.createTrainedClusterer(filename, tMCOD.kOption.getValue(), tMCOD.radiusOption.getValue(), tMCOD.windowSizeOption.getValue(), numInstances, threadNum, stop);
        }
        catch(IllegalThreadStateException e){
            System.out.println("IllegalThreadStateException: " + e.getMessage());
        }
    }

    public void threadStop(){
        stop.set(true);
    }
}
