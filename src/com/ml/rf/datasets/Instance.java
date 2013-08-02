package com.ml.rf.datasets;

public class Instance {
    private double[] featureVector;
    /** label index  */
    private int labelIndex;   
    
    public Instance(double[] featureVector, int labelIndex){
        this.featureVector = featureVector;
        this.labelIndex = labelIndex;
    }
    
    public double[] getFeatureVector(){
        return this.featureVector;
    }
    
    public int getLabelIndex(){
        return labelIndex;
    }
    
}
