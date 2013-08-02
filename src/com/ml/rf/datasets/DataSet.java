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
    // total number of class labels. Each label is an integer. 
    private int numOfLabels;
   
    public DataSet(List<Instance> trainingData){
        this.trainingData = trainingData;
        this.numOfInstance = trainingData.size();
        this.numOfFeatures = trainingData.get(0).getFeatureVector().length;
        
        Set<Integer> labelSet = new HashSet<Integer>();
        for (int i = 0; i < trainingData.size(); i++)
            labelSet.add(trainingData.get(i).getLabelIndex());
        this.numOfLabels = labelSet.size();
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
    
    public int getNumOfLabels(){
        return numOfLabels;
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
