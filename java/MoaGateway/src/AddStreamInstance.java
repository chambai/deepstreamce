import java.util.ArrayList;
import java.util.Arrays;
import java.util.UUID;

public class AddStreamInstance{
    private String Id;
    private double[] attributeValues;
    private String[] attributeLabels;
    private double classValue;

    AddStreamInstance(String id, double[] attributeValues, String[] attributeLabels, double classValue)
    {
        this.Id = id;
        this.attributeValues = attributeValues;
        this.attributeLabels = attributeLabels;
        this.classValue = classValue;
    }

    AddStreamInstance(String id, double[] attributeValues, double classValue)
    {
        this.Id = id;
        this.attributeValues = attributeValues;
        this.attributeLabels = this.setAttributeLabels(attributeValues);
        this.attributeLabels = new String[]{};
        this.classValue = classValue;
    }

    AddStreamInstance(String id, double[] attributeValues, String[] attributeLabels)
    {
        this.Id = id;
        this.attributeValues = attributeValues;
        this.attributeLabels = attributeLabels;
        this.classValue = -1;
    }

    AddStreamInstance(String id, double[] attributeValues)
    {
        this.Id = id;
        this.attributeValues = attributeValues;
        this.attributeLabels = this.setAttributeLabels(attributeValues);
        this.classValue = -1;
    }

    private String[] setAttributeLabels(double[] attributeValues){
        // set the attribute labels to increment if they have not been defined
        var attributeLabels = new String[this.attributeValues.length];
        for(int i=0; i<attributeValues.length; i++){
            attributeLabels[i] = "attr" + i;
        }
        return attributeLabels;
    }

    public double[] getAttributeValues(){
        return this.attributeValues;
    }

    public String[] getAttributeLabels(){
        return this.attributeLabels;
    }

    public ArrayList<String> getAttributeLabelsAsList(){
        return new ArrayList<String>(Arrays.asList(this.attributeLabels));
    }

    public double getClassValue(){
        return this.classValue;
    }

    public double[] getAllValues(){
        double result[] = new double[attributeValues.length + 1];
        for(int i=0; i<attributeValues.length + 1; i++){
            if(i < attributeValues.length)
                result[i] = attributeValues[i];
            else
                result[i] = this.classValue;
        }
        return result;
    }

    @Override
    public String toString(){
        String result = "";

        for(var value : attributeValues){
            result += value + ", ";
        }

        if(this.classValue != -1) {
            result += this.classValue;
        }

        return result;
    }

    public String getId(){
        return this.Id;
    }

    public void setId(String id){
        this.Id = id;
    }
}