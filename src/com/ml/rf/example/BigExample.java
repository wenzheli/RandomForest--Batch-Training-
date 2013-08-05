package com.ml.rf.example;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ml.rf.classifier.RandomForest;
import com.ml.rf.datasets.DataSet;
import com.ml.rf.datasets.Instance;

public class BigExample {
    
    public Map<String, Integer> labelIndexMap = new HashMap<String, Integer>();
    public Map<Integer, String> indexLabelMap = new HashMap<Integer, String>();
    
    public static final String TRAIN_DATA = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data";
    public static final String TRAIN_LABLE = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels";
    
    public static final String TEST_DATA = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data";
    public static final String TEST_LABEL = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels";
    
    public static void main(String[] args) throws IOException{
        BigExample example = new BigExample();
        List<double[]> data = example.getData(TRAIN_DATA);
        List<Integer> labels = example.getLabel(TRAIN_LABLE);
        
        RandomForest rf = new RandomForest(new DataSet(data, labels));
      
        List<double[]> testData = example.getData(TEST_DATA);
        List<String> correctLabels = example.getLabelOnly(TEST_LABEL);
        
        int correct = 0;
        int total = 0;
        for (int i = 0; i < testData.size(); i++){
            double[] featureVector = testData.get(i);
            int correctLabel = rf.predictLabel(featureVector);
            System.out.println(correctLabel);
            String strLabel = example.indexLabelMap.get(correctLabel);
            if (correctLabels.get(i).equals(strLabel))
                correct++;
            total++;
        }
        
        System.out.println(correct);
        System.out.println(total); 
    }
    
    
    /**
     * Create a dataset from URL. The data set comes from 
     * http://archive.ics.uci.edu/ml/datasets/Madelon
     * 
     */
    private List<double[]> getData(String data) throws IOException{
        URL dataUrl = new URL(data);
        BufferedReader in = new BufferedReader(
        new InputStreamReader(dataUrl.openStream()));
        int cnt= 0;
        List<double[]> samples = new ArrayList<double[]>();
        String inputLine;
        while ((inputLine = in.readLine()) != null){
            String[] features = inputLine.split("\\s+");
            double[] sample = new double[features.length];
            for (int i = 0; i < features.length; i++){
                sample[i] = Double.parseDouble(features[i]);
            }
            samples.add(sample);
        }
        in.close();
              
        return samples;
    }
    
    private List<Integer> getLabel(String label) throws IOException{
        URL labelUrl = new URL(label);
        List<Integer> labels = new ArrayList<Integer>();
        int cnt = 0;
        BufferedReader in = new BufferedReader(
                new InputStreamReader(labelUrl.openStream()));
        int idx = 0;
        String inputLine;
        while ((inputLine = in.readLine()) != null){
            if (!labelIndexMap.containsKey(inputLine)){
                labelIndexMap.put(inputLine, idx);
                indexLabelMap.put(idx, inputLine);
                idx++;
            }
            labels.add(labelIndexMap.get(inputLine));
        }
        in.close();   
        
        return labels;
    }
    
    private List<String> getLabelOnly(String label) throws IOException{
        URL labelUrl = new URL(label);
        List<String> labels = new ArrayList<String>();
        
        BufferedReader in = new BufferedReader(
                new InputStreamReader(labelUrl.openStream()));
     
        String inputLine;
        while ((inputLine = in.readLine()) != null){
          
            labels.add(inputLine);
        }
        in.close();   
        
        return labels;
    }
}
