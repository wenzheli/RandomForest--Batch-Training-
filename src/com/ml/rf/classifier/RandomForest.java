package com.ml.rf.classifier;

import java.util.ArrayList;
import java.util.List;

import com.ml.rf.datasets.DataSet;
import com.ml.rf.datasets.Instance;
import com.ml.rf.utils.SamplerUtils;

/**
 * Random forest classifier, which contains multiple decision trees. 
 * The process of building random forest is :
 * 
 * (a) Draw a bootstrap sample Z* of size N from the training data
 * (b) Grow a random-forest tree T_{b} to the bootstrap data , by recursively
 *     repeating the following steps for each terminal node of the tree, until 
 *     the minimum node size MIN_TREE_SIZE is reached. 
 *     (i) Select m variables at random from the total feature set. 
 *     (ii) Pick the best variable/split-point among the m
 *     (iii) Split the node into left and right child. 
 *  
 * There are two random properties involves when building random forest. (i) When
 * building each decision tree, we built it based on bootstrap. (ii) For 
 * every node for decision tree, we first randomly select subset of features, 
 * and then select the best feature among them using information gain. 
 * 
 * Reference:
 *  (1)   Trevor Hastie, Robert Tibshirani, Jerome Friedman. The element of statistical
 *        learning: data mining, inference and prediction. Springer.
 *  (2)   Leo Breiman. Random Forest, UC Berkeley. 2001 
 *        http://oz.berkeley.edu/~breiman/randomforest2001.pdf
 *  (3)   ID3 algorithm  http://en.wikipedia.org/wiki/ID3_algorithm 
 *      
 * @author wenzhe
 *
 */
public class RandomForest {
   
    private int numTrees; 
    private List<DecisionTree> decisionTrees;
    
    // size of sampling for each bootstrap step. 
    private int sampleFeatureSize;
    
    // minimum number of samples for each node. If reached the minimum, we just make it as
    // leaf node without further splitting. 
    public static final int TREE_MIN_SIZE = 1;
    
    private DataSet dataset;
      
    public RandomForest(DataSet dataset){
        this.dataset = dataset;
        numTrees = 100;  // TODO need a adaptable default value. 
        decisionTrees = new ArrayList<DecisionTree>(numTrees);
        this.sampleFeatureSize = 20; // TODO need a adaptable default value.
        createRandomForest();
    }
    
    public RandomForest(DataSet dataset,int numTrees, int sampleFeatureSize){
        this.dataset = dataset;
        this.numTrees = numTrees;
        this.sampleFeatureSize = sampleFeatureSize;
        decisionTrees = new ArrayList<DecisionTree>(numTrees);
        createRandomForest();
    }
    
    /**
     * Build random forest. TODO: multi-threading/distributed computing.  
     */
    public void createRandomForest(){
        for (int i = 0; i < numTrees; i++){
            System.out.println("creating " + i + "th tree");
            DecisionTree dt = new DecisionTree();
            dt.setTreeMinSize(TREE_MIN_SIZE);
            dt.buildTree(getBootStrapData(),sampleFeatureSize);
           
            decisionTrees.add(dt);
        }
    }
    
  
    /**
     * Get the predicted label for given feature vector. 
     * 
     * @param featureVector  the input feature vector. 
     * @return     predicted label. 
     */
    public int predictLabel(double[] featureVector){
        double[] dist = predictDist(featureVector);
        int maxLabel = 0;
        double maxProb = 0;
        for (int i = 0; i < dist.length; i++){
            if (dist[i] > maxProb){
                maxProb = dist[i];
                maxLabel = i;
            }
        }
        return maxLabel;
    }
    
    /**
     * Get the prediction for the input feature vector.  Basically, it iterate 
     * through each decision tree, and get prediction from each of them. Then aggregate
     * those predictions. 
     * 
     * @param featureVector      the query input feature vector. 
     * @return                   prediction, which is probability distribution for different
     *                              labels. 
     */
    public double[] predictDist(double[] featureVector){
        int totalNumLabels = 2;
        double[] finalPredict = new double[totalNumLabels];
        // iterate through each decision tree, and make prediction. 
        for (int i = 0; i < numTrees; i++){
           double[] predict = decisionTrees.get(i).predictDist(featureVector);
           for (int j = 0; j < totalNumLabels; j++){
               finalPredict[j] += predict[j];
           }
        }
        
        for (int i = 0; i < totalNumLabels; i++){
            finalPredict[i] = finalPredict[i]/numTrees;
        }
        
        return finalPredict;
    }
    
    
    /**
     * Get bootstrap dataset, with replacement. 
     */
    private DataSet getBootStrapData(){
        int[] indexs = SamplerUtils.bootStrap(dataset.getNumOfInstance());
        List<Instance> bootStrapSamples = new ArrayList<Instance>();
        for (int i = 0; i < indexs.length; i++){
            bootStrapSamples.add(dataset.getInstance(indexs[i]));
        }
        
        return new DataSet(bootStrapSamples);
    }
}
