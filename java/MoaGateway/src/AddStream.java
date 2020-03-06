import java.util.ArrayList;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.streams.clustering.ClusteringStream;
import moa.tasks.TaskMonitor;

/**
 * Provides a simple input stream for in memory instances. Adds if necessary a class
 * attribute with an identical default value.
 *
 */
public class AddStream extends ClusteringStream {

    private static final long serialVersionUID = 1L;

    protected ArrayList<AddStreamInstance> addStreamInstances;

    protected ArrayList<String> attributeNames;

    protected boolean hasMoreInstances;

    protected Instances dataset;

    protected IdInstanceExample lastInstanceRead;

    protected int numInstancesRead;

    private ArrayList<String> classLabels;

    //protected int numAttributes;

    /**
     * Creates a simple ClusteringStream for csv files. Adds if necessary a
     * class attribute with an identical default value.
     *
     */
    public AddStream(ArrayList<String> attributeNames, ArrayList<String> classLabels) {
        this.attributeNames = attributeNames;
        this.classLabels = classLabels;
        //this.numAttributes = this.attributeNames.size();
        restart();
    }

    public AddStream(ArrayList<String> attributeNames) {
        this.attributeNames = attributeNames;
        this.classLabels = new ArrayList<String>();
        //this.numAttributes = this.attributeNames.size();
        restart();
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.options.AbstractOptionHandler#getPurposeString()
     */
    @Override
    public String getPurposeString() {
        return "A stream read from in memory instances.";
    }

    /*
     * (non-Javadoc)
     *
     * @see
     * moa.options.AbstractOptionHandler#prepareForUseImpl(moa.tasks.TaskMonitor
     * , moa.core.ObjectRepository)
     */
    @Override
    public void prepareForUseImpl(TaskMonitor monitor,
                                  ObjectRepository repository) {
        restart();
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#getHeader()
     */
    @Override
    public InstancesHeader getHeader() {
        return new InstancesHeader(this.dataset);
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#estimatedRemainingInstances()
     */
    @Override
    public long estimatedRemainingInstances() {
        // N/A as we do not know how many instances will be added
        return -1;
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#hasMoreInstances()
     */
    @Override
    public boolean hasMoreInstances() {

        return !this.addStreamInstances.isEmpty();
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#nextInstance()
     */
    @Override
    public InstanceExample nextInstance() {
        // get next instance from the list and set it into dataset
        boolean isSuccess = this.processNextDataInstance();
        if(isSuccess)
        {
            this.numInstancesRead++;
        }

        IdInstanceExample prevInstance = this.lastInstanceRead;

        return prevInstance.getInstanceExample();
    }

    public IdInstanceExample nextIdInstance() {
        // get next instance from the list and set it into dataset
        boolean isSuccess = this.processNextDataInstance();
        if(isSuccess)
        {
            this.numInstancesRead++;
        }

        IdInstanceExample prevInstance = this.lastInstanceRead;

        return prevInstance;
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#isRestartable()
     */
    @Override
    public boolean isRestartable() {
        return true;
    }

    /*
     * (non-Javadoc)
     *
     * @see moa.streams.InstanceStream#restart()
     */
    @Override
    public void restart() {
        this.addStreamInstances = new ArrayList<AddStreamInstance>();
        this.numAttsOption = null;
        this.dataset = null;
        this.lastInstanceRead = null;
        this.numInstancesRead = 0;
        this.hasMoreInstances = false;

        // Get the number of attributes from the instance that is passed in dict.value.inputdata (number of layers)
        // on construction, pass in number of attributes (number of layers) and store in this.numAttributes
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (String attName : this.attributeNames) {
            attributes.add(new Attribute(attName));
        }

        // Add class labels to the end of the attributes
        if(this.classLabels.size() == -1)
        {
           //this.classIndexOption.unset();
        }
        attributes.add(new Attribute("class", classLabels));

        // Set the Instances
        this.dataset = new Instances("Add Stream Instances",
                attributes, 0);

        // Set the class index of the Instances
        this.dataset.setClassIndex(this.attributeNames.size());

        // Set the numAttsOption - this must be in the super class
        numAttsOption = new IntOption("numAtts", 'a', "",
                this.attributeNames.size());
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        sb.append("AddStream: ");
        sb.append("Adds in-memory instances to a stream");
    }

    public void addInstance(String id, ArrayList<Double> data, ArrayList<String> labels){

        double[] target = new double[data.size()];
        String[] targetLabels = new String[data.size()];
        for (int i = 0; i < target.length; i++) {
            target[i] = data.get(i);
            targetLabels[i] = labels.get(i);
        }

        var instance = new AddStreamInstance(id, target, targetLabels);
        this.addInstance(instance);
    }

    public void addInstanceStr(String id, String dataPickle, String labelPickle){

        System.out.println(dataPickle);
        System.out.println(labelPickle);

        ArrayList<Double> target = new ArrayList<Double>();
        ArrayList<String> targetLabels = new ArrayList<String>();
        // deserailize the data
        target.add(0.10987);
        target.add(0.10787);
        target.add(0.10587);
        target.add(0.10587);
        targetLabels.add("1");
        targetLabels.add("2");
        targetLabels.add("3");
        targetLabels.add("4");

        this.addInstance(id, target, targetLabels);
    }

    public void addInstance(String id, ArrayList<Double> data){

        double[] target = new double[data.size()];
        for (int i = 0; i < target.length; i++) {
            target[i] = data.get(i);
        }

        var instance = new AddStreamInstance(id, target);
        this.addInstance(instance);
    }

    public void addInstance(AddStreamInstance dataInstance)
    {

        // Check the number of values in the instance matches the number of attributes specified for this stream
        if(dataInstance.getAttributeValues().length != this.attributeNames.size())
            throw new
                    IllegalArgumentException("The number of values provided must match the number of attributes specified. Data values: " + dataInstance.getAttributeValues().length + ", Num Attr:" + (this.attributeNames.size()-1));

        // Add this instance to the list of instances to be processed
        this.addStreamInstances.add(dataInstance);
    }

    private boolean processNextDataInstance()
    {
        boolean hasNextInstance = false;

        // Get the first entry in the array list of data instances
        if(this.addStreamInstances.size() > 0) {
            AddStreamInstance dataInstance = this.addStreamInstances.get(0);

            // Create an Instance Example of Instance type DenseInstance to store the values and store in
            // last instance read
            this.lastInstanceRead = new IdInstanceExample(dataInstance);
            //this.lastInstanceRead = (IdInstanceExample) new InstanceExample(new DenseInstance(1, dataInstance.getAllValues()));
            // set this instance into the dataset (Instances)
            this.lastInstanceRead.getInstanceExample().getData().setDataset(this.dataset);

            // Remove instance from the ArrayList
            this.addStreamInstances.remove(0);

            hasNextInstance = true;
        }
        else
        {
            this.lastInstanceRead = null;
        }

        return hasNextInstance;
    }


}