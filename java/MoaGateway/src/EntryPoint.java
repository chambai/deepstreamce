import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import com.yahoo.labs.samoa.instances.Instance;
import py4j.GatewayServer;
import moa.clusterers.outliers.MCOD.MCOD;
import moa.clusterers.outliers.MyBaseOutlierDetector;


public class EntryPoint {

    static ConcurrentHashMap<String, List<MCOD>> mapTrainedMcods = new ConcurrentHashMap<String, List<MCOD>>();
    static ConcurrentHashMap<String, ClusterCreatorThread> mapClusterCreatorThreads = new ConcurrentHashMap<String, ClusterCreatorThread>();
    boolean stop = false;


    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new EntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
        var runtime = Runtime.getRuntime();
        System.out.println("maxMemory: " + runtime.maxMemory());
        System.out.println("totalMemory: " + runtime.totalMemory());
        System.out.println("freeMemory: " + runtime.freeMemory());
    }


    public MCOD Moa_Clusterers_Outliers_MCOD_New(){

        var mcod = new MCOD();
        mcod.Init();
        return mcod;
    }


    public String Moa_Clusterers_Outliers_MCOD_GetParameters(MCOD mcod){
        var params = new ArrayList<String>();

        var param = "k = " + mcod.kOption.getValue();
        params.add(param);

        param = "radius = " + mcod.radiusOption.getValue();
        params.add(param);

        param = "window size = " + mcod.windowSizeOption.getValue();
        params.add(param);


        return String.join(",", params);
    }

    public String Moa_Clusterers_Outliers_MCOD_SetParameters(MCOD mcod, int k, double radius, int windowSizeOption){
        mcod.kOption.setValue(k);
        mcod.radiusOption.setValue(radius);
        mcod.windowSizeOption.setValue(windowSizeOption);

        return Moa_Clusterers_Outliers_MCOD_GetParameters(mcod);
    }


    public ArrayList<String> Moa_Clusterers_Outliers_MCOD_addAndAnalyse(MCOD tMcod, Instance instance, String csvTrainFilename, int numThreads){

        var results = new ArrayList<String>();

        // get mcod from map
        MCOD iMcod = null;
        while(iMcod == null){
            try {
                iMcod = this.getTrainedClusterer(iMcod, csvTrainFilename, numThreads);
            }
            catch(IndexOutOfBoundsException e){
                // there are no trained MCODs ready
                System.out.println("No trained MCODs ready: " + e.getMessage());
                try{
                    Thread.sleep(500);
                }
                catch(InterruptedException ie){
                    System.out.println("Thread sleep failed: " + ie.getMessage());
                }
            }
        }

        System.out.println("Trained MCOD hash code: " + tMcod.hashCode());
        System.out.println("Copied MCOD hash code: " + iMcod.hashCode());

        var statsBefore = iMcod.getStatistics();
        iMcod.processNewInstanceImpl(instance);
        var statsAfter = iMcod.getStatistics();

        results.addAll(this.Moa_Clusterers_Outliers_MCOD_GetResults(iMcod, instance));
        results.add("B:"+ statsBefore);
        results.add("A:"+ statsAfter);

        return results;
    }

    private MCOD getTrainedClusterer(MCOD iMcod, String csvTrainFilename, int numThreads){

        String key = csvTrainFilename;
        if (mapTrainedMcods.size() > 0) {
            if (mapTrainedMcods.containsKey(key)) {
                iMcod = mapTrainedMcods.get(key).remove(0);
            } else {

            }
        }

        return iMcod;
    }


    public void createTrainedClusterer(String filename, int k, double radius, int windowSizeOption, int numUnseenInstances, int threadNum, AtomicBoolean stop){

        var csvData = this.getDataFromCsv(filename);
        // Create the stream for use with the CSV data
        var attributes = csvData.getInstances().get(0).getAttributeLabelsAsList();

            var stream = this.Moa_Streams_AddStream_New(attributes);
            stream.prepareForUse();

            // Add the instances to the stream
            this.Moa_Clusterers_Outliers_Mcod_AddInstancesToStream(stream, csvData.getInstances());

            // create the clusterer
            var mcod = this.Moa_Clusterers_Outliers_MCOD_New();
            this.Moa_Clusterers_Outliers_MCOD_SetParameters(mcod, k, radius, windowSizeOption);
            mcod.setModelContext(stream.getHeader());
            mcod.prepareForUse();

            // populate the clusterer with stream data
            while (stop.get() == false && stream.hasMoreInstances()) {

                var newInstEx = stream.nextIdInstance();
                if (newInstEx != null) {
                    String instId = newInstEx.getIdAsString();
                    var newInst = newInstEx.getInstanceExample().getData();

                    try {
                        mcod.processNewInstanceImpl(newInst);
                    } catch (Exception e) {
                        System.out.println("Instance could not be processed:\n" + e);
                    }
                }
            }

            // Add MCOD to Map
            var key = filename;
            if (!mapTrainedMcods.containsKey(key)) {
                mapTrainedMcods.put(key, Collections.synchronizedList(new ArrayList<>()));
            }

            mapTrainedMcods.get(key).add(mcod);
    }

    public void Moa_Clusterers_Outliers_MCOD_processNewInstanceImplTrain(MCOD tMcod, Instance instance){

        tMcod.processNewInstanceImpl(instance);
    }

    public void CreateTrainedClusterers(MCOD tMCOD, String filename, int numInstances, int numThreads){

        try {
            ThreadExecutor tExec = new ThreadExecutor(this, tMCOD, filename, numInstances, numThreads);
            tExec.execute();
            System.out.println("Num MCOD class map: " + mapTrainedMcods.size());
            for(var entrySet : mapTrainedMcods.entrySet()){
                System.out.println("Num MCODs for class " + entrySet.getKey() + ": " + entrySet.getValue().size());
            }
        }
        catch(ExecutionException e){
            System.out.println("ExecutionException when creating MCODs" + e.getMessage());
        }
        catch(InterruptedException e){
            System.out.println("InterruptedException when creating MCODs" + e.getMessage());
        }
    }

    public void StopCreatingTrainedClusterers(){

        for(var entrySet : mapClusterCreatorThreads.entrySet()){
            entrySet.getValue().threadStop();
        }
        mapClusterCreatorThreads.clear();
    }

    private boolean createThread(MCOD tMCOD, String filename, int numInstances, int threadNum) {
        boolean threadStarted = false;
        String key = filename + "_" + threadNum;
        if(!mapClusterCreatorThreads.containsKey(key)) {
            // if the thread does not exist, create and start it
            ClusterCreatorThread t = new ClusterCreatorThread(this, tMCOD, filename, numInstances, threadNum);
            t.start();
            mapClusterCreatorThreads.put(key, t);
            threadStarted = true;
            try {
                Thread.sleep(5000);
            }
            catch(InterruptedException ie){
                System.out.println("Thread sleep failed: " + ie.getMessage());
            }
        }
        else{
            System.out.println("Thread " + key + " exists");
            // if it's not alive, start it
            if(!mapClusterCreatorThreads.get(key).isAlive())
            {
                mapClusterCreatorThreads.get(key).start();
                threadStarted = true;
            }
        }
        return threadStarted;
    }

    public ArrayList<String> Moa_Clusterers_Outliers_MCOD_GetResults(MCOD mcod, Instance instance){
        ArrayList<String> results = new ArrayList<>();
        try {
            results = new ArrayList<>();
            //results.addAll(Arrays.asList(this.Moa_Clusterers_Outliers_MCOD_GetCenter(mcod)));
            results.add(this.Moa_Clusterers_Outliers_MCOD_GetOutlierResult(mcod, instance));
            results.addAll(this.Moa_Clusterers_Outliers_MCOD_GetInlierStatus(mcod, instance));
        }
        catch(Exception exc){
            results.add("ERROR," + exc.toString());
        }

        String[] resultArray = new String[results.size()];
        resultArray = results.toArray(resultArray);
        return results;
    }

    public ArrayList<String> Moa_Clusterers_Outliers_MCOD_GetInlierStatus(MCOD mcod, Instance instance)
    {
        // just returns all of the instances
        var clusterResult = mcod.getClusteringResult();

        ArrayList<String> result = new ArrayList<>();

        if(clusterResult == null) {
            result.add("NODATA,CENTER,null cluster result");
        }
        else if(clusterResult.size() == 0){
            result.add("NODATA,CENTER,empty cluster result");
        }
        else{
            result.add("DATA,CENTER,NumClusters=" + clusterResult.size());
        }

        return result;
    }

    public String Moa_Clusterers_Outliers_MCOD_GetOutlierResult(MCOD mcod, Instance instance)
    {
        var outliers = mcod.getOutliersResult();


        if(outliers == null)
            return "NODATA,OUTLIER,NULL";   // null outlier result
        else{
            if(outliers.size() > 0) {
                if (this.isInstanceInOutliers(outliers, instance))
                    return "DATA,OUTLIER,OUTLIER"; // this instance is outlier
                else
                    // outliers reported, this instance is not an outlier. it could be an inlier or not in a micro cluster
                    return "DATA,OUTLIER,NOT_OUTLIER";
            }
            else
                return "DATA,OUTLIER,NO_OUTLIERS_REPORTED"; // no outliers reported
        }
    }

    private boolean isInstanceInOutliers(Vector<MyBaseOutlierDetector.Outlier> outliers, Instance inst){

        var isMatch = false;
        for (var outlier : outliers) {
            isMatch = this.compareInstances(outlier.inst, inst);
            if(isMatch) {
                isMatch = true;
                break;
            }
        }
        return isMatch;
    }

    private boolean compareInstances(Instance inst1, Instance inst2)
    {
        var isMatch = true;
        var inst1Arry = inst1.toDoubleArray();
        var inst2Arry = inst2.toDoubleArray();
        if(inst1Arry.length == inst2Arry.length)
        {
            for(int i=0; i<inst1Arry.length; i++ )
            {
                if(inst1Arry[i] != inst2Arry[i]) {
                    isMatch = false;
                    break;
                }

            }
        }

        return isMatch;
    }

    public AddStream Moa_Streams_AddStream_New(ArrayList<String> attributeList) {
        //return new AddStream(attributeList, classLabels);
        return new AddStream(attributeList);
    }


    public void Moa_Clusterers_Outliers_Mcod_AddInstancesToStream(AddStream addStream, List<AddStreamInstance> dataStreamInstances){
        addStream.addStreamInstances.addAll(dataStreamInstances);
    }

    public void Moa_Clusterers_Outliers_Mcod_AddCsvDataToStream(AddStream addStream, String file, ArrayList<String> ids){
        var csvData = getDataFromCsv(file, ids);
        Moa_Clusterers_Outliers_Mcod_AddInstancesToStream(addStream, csvData.getInstances());
    }

    public class CsvData{
        private ArrayList<AddStreamInstance> instances;
        private ArrayList<String> labels;

        CsvData(ArrayList<AddStreamInstance> instances, ArrayList<String> labels){
            this.instances = instances;
            this.labels = labels;
        }

        ArrayList<AddStreamInstance> getInstances(){
            return this.instances;
        }

        ArrayList<String> getLabels(){
            return this.labels;
        }
    }

    public CsvData getDataFromCsv(String file) {

        ArrayList<String> dataList = new ArrayList<String>();
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(file));
            var row = "";
            var index = 0;
            while ((row = csvReader.readLine()) != null) {
                // Do not store the first row of the csv file as it is a label
                if(index > 0)
                    dataList.add(row);
                index++;
            }
        }
        catch(java.io.IOException ioExc)
        {
            System.out.println("File IO exception: " + ioExc.toString());
        }
        dataList.removeAll(Arrays.asList("", null));

        var labels = new ArrayList<String>();
        ArrayList<AddStreamInstance> dataInstances = new ArrayList<AddStreamInstance>();
        for(var dataStr : dataList){

            var instanceStrArray = dataStr.split(",");

            double[] target = new double[instanceStrArray.length];
            for (int i = 0; i < target.length; i++) {
                var value = instanceStrArray[i].replace("\"","");
                if(i == target.length-1){
                    labels.add(value);
                }
                else
                {
                    target[i] = Double.parseDouble(value);
                }
            }

            var dataStreamInstance = new AddStreamInstance(UUID.randomUUID().toString(), target);
            dataInstances.add(dataStreamInstance);
        }

        return new CsvData(dataInstances, labels);
    }

    public CsvData getDataFromCsv(String file, ArrayList<String> ids) {

        ArrayList<String> dataList = new ArrayList<String>();
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(file));
            var row = "";
            var index = 0;
            while ((row = csvReader.readLine()) != null) {
                // Do not store the first row of the csv file as it is a label
                if(index > 0)
                    dataList.add(row);
                index++;
            }
        }
        catch(java.io.IOException ioExc)
        {
            System.out.println("File IO exception: " + ioExc.toString());
        }
        dataList.removeAll(Arrays.asList("", null));

        var labels = new ArrayList<String>();
        ArrayList<AddStreamInstance> dataInstances = new ArrayList<AddStreamInstance>();
        for(int d=0; d<dataList.size(); d++){
            var dataStr = dataList.get(d);
            var instanceStrArray = dataStr.split(",");

            double[] target = new double[instanceStrArray.length];
            for (int i = 0; i < target.length; i++) {
                var value = instanceStrArray[i].replace("\"","");
                if(i == target.length-1){
                    labels.add(value);
                }
                else
                {
                    target[i] = Double.parseDouble(value);
                }
            }

            var dataStreamInstance = new AddStreamInstance(ids.get(d), target);
            dataInstances.add(dataStreamInstance);
        }

        return new CsvData(dataInstances, labels);
    }

}
