package com.ml.rf.datasets;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Dataset model contains training data, and some properties of training set. 
 * 
 * @author wenzhe
 * 
 */
public class DataSet {
    
    private List<Instance> trainingData = new ArrayList<Instance>();
    
    // total number of instance in this training set
    private int numOfInstance;
    // total number of features. 
    private int numOfFeatures;
   
   
    public DataSet(List<Instance> trainingData){
        this.trainingData = trainingData;
        init();
    }
    
    public DataSet(List<double[]> featureVectors, List<Integer> labels){
        trainingData = new ArrayList<Instance>();
        for (int i = 0; i < featureVectors.size(); i++){
            trainingData.add(new Instance(featureVectors.get(i), labels.get(i)));
        }
        init();
    }
     
    private void init(){
        this.numOfInstance = trainingData.size();
        this.numOfFeatures = trainingData.get(0).getFeatureVector().length;
    }
    
    public List<Instance> getTrainingData(){
        return trainingData;
    }
    
    public int getNumOfInstance(){
        return numOfInstance;
    }
    
    public int getNumOfFeatures(){
        return numOfFeatures;
    }
      
    public List<Integer> getLabels(){
        List<Integer> labels = new ArrayList<Integer>();
        for (Instance instance : trainingData){
            labels.add(instance.getLabelIndex());
        }
        
        return labels;
    }
    
    /*
     * return ith instance. 
     */
    public Instance getInstance(int i){
        return trainingData.get(i);
    }
}
