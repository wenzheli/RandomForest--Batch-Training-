package com.ml.rf.example;

import java.util.ArrayList;
import java.util.List;

import com.ml.rf.classifier.RandomForest;
import com.ml.rf.datasets.DataSet;

public class SimpleExample {
    public static void main(String[] args){
        
        List<double[]> train = new ArrayList<double[]>();
        train.add(new double[]{1,2});
        train.add(new double[]{1,3});
        train.add(new double[]{1,4});
        train.add(new double[]{1,5});
        train.add(new double[]{2,3});
        train.add(new double[]{2,4});
        train.add(new double[]{2,5});
        train.add(new double[]{3,4});
        train.add(new double[]{5,1});
        train.add(new double[]{5,2});
        train.add(new double[]{5,3});
        train.add(new double[]{5,4});
        train.add(new double[]{6,1});
        train.add(new double[]{6,2});
        train.add(new double[]{6,3});
        train.add(new double[]{4,2});
        
        List<Integer> label = new ArrayList<Integer>();
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(0);
        label.add(1);
        label.add(1);
        label.add(1);
        label.add(1);
        label.add(1);
        label.add(1);
        label.add(1);
        label.add(1);
           
        List<double[]> test = new ArrayList<double[]>();
        test.add(new double[]{3,5});   // 0
        test.add(new double[]{1,10});  // 0
        test.add(new double[]{1.5,5}); // 0
        test.add(new double[]{3.6,1}); // 1
        test.add(new double[]{6,5});   // 1
        test.add(new double[]{7,5});   // 1
        
        RandomForest rf = new RandomForest(new DataSet(train, label), 100, 2);
        
        for (double[] testSample: test){
            System.out.println(rf.predictLabel(testSample));
        } 
    }
}
