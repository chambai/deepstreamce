import com.yahoo.labs.samoa.instances.DenseInstance;
import moa.core.InstanceExample;

public class IdInstanceExample {

    private String id;
    private InstanceExample instanceExample;

    public IdInstanceExample(AddStreamInstance addStreamInstance)
    {
        this.instanceExample = new InstanceExample(new DenseInstance(1, addStreamInstance.getAllValues()));
        this.id = addStreamInstance.getId();
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId(){
        return this.id;
    }

    public String getIdAsString(){
        return this.id.toString();
    }


    public InstanceExample getInstanceExample(){
        return this.instanceExample;
    }
}
