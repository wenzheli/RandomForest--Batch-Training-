package com.ml.rf.classifier;

import java.util.ArrayList;
import java.util.List;

import com.ml.rf.datasets.DataSet;
import com.ml.rf.datasets.Instance;
import com.ml.rf.nodes.DecisionNode;
import com.ml.rf.nodes.LeafNode;
import com.ml.rf.nodes.TreeNode;
import com.ml.rf.utils.EntropyUtils;
import com.ml.rf.utils.SamplerUtils;


public class DecisionTree {
    
    // root of the decision tree. 
    private TreeNode root;
   
    // for each node, we randomly select subset of features to consider for splitting
    // by default, we set the size as square root of total number of features.  
    private int sampleFeatureSize;
    
    private int numLabels = 2;
    
    // minimum size of subtree, this value can be used as condition for termination.
    // by default, we set the size as 5. 
    private int minTreeSize = 10;
    
    public DecisionTree(){ 
    }
    
    public void buildTree(DataSet dataset, int sampleFeatureSize){
        this.sampleFeatureSize = sampleFeatureSize;
        // create a root node, select the label with the largest information gain
        root = build(dataset);
    }
    
    /**
     * predict the distribution for given feature vector.  
     * @param featureVector        the query feature vector. 
     * @return                     predicted distribution. 
     */
    public double[] predictDist(double[] featureVector){
        return predict(root, featureVector);
    }  
    
    
    /**
     * predict the label for given feature vector. 
     * @param featureVector     the query feature vector
     * @return                  predicted label. 
     */
    public int predictLabel(double[] featureVector){
        double[] dist = predict(root, featureVector);
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
     * make prediction for the input feature vector. The recursive function stops
     * when we encounter leaf node, which contains the probability distribution for 
     * different labels.  
     * 
     * @param node            current node we are searching for . 
     * @param featureVector   the feature vector. 
     * @return                probability distribution. 
     */
    private double[] predict(TreeNode node, double[] featureVector){
        // if leaf node, then just return the distribution. 
        if (node instanceof LeafNode)
            return ((LeafNode) node).getLabelProbDist();
        
        // if current node is decision node, then go to left child or right child. 
        int featureIdx = ((DecisionNode)node).getSplittingFeatureIndex();
        double splittingValue = ((DecisionNode)node).getSplittingValue();
        double value = featureVector[featureIdx];
        if (value < splittingValue)
            return predict(((DecisionNode)node).getLeftChild(), featureVector);
        else
            return predict(((DecisionNode)node).getRightChild(), featureVector);
    }
    
    /**
     * build the decision tree recursively. 
     */
    private TreeNode build(DataSet dataset){
        // create a new leaf node if either of condition met. 
        if (dataset.getNumOfInstance() < minTreeSize || hasSameLabel(dataset.getLabels())){
            TreeNode leafNode = new LeafNode(numLabels, dataset.getLabels());
            return leafNode;
        }
        
        // sub-sample the attributes. 
        int[] selectedFeatureIndexs = SamplerUtils.randSample(dataset.getNumOfFeatures(), sampleFeatureSize);
        
        // select the best feature based on information gain
        int bestFeatureIndex = getBestFeatureIndex(selectedFeatureIndexs, dataset);
        
        // for numerical attribute, we  create left and right child. 
        return createDecisionNode(bestFeatureIndex, dataset);
    }
    
     
    /**
     * get the best feature, by calculating the information gain. 
     * Return the index of best feature. 
     */
    private int getBestFeatureIndex(int[] candiateFeatureIndexs, DataSet dataset){
        double maxInfoGain = Double.MIN_VALUE;
        int bestFeatureIndex = 0;
        double entropy = EntropyUtils.getEntropy(dataset.getLabels());
        
        for (int i = 0; i < candiateFeatureIndexs.length; i++){
            int featureIndex = candiateFeatureIndexs[i];
            double infoGain = getInformationGain(entropy, featureIndex, dataset);
            if (infoGain > maxInfoGain){
                maxInfoGain = infoGain;
                bestFeatureIndex = i;
            }
        }
        return bestFeatureIndex;
    }
    
    
    /**
     * Calculate the information gain for certain feature. The calculation considers two cases, 
     * one for categorical attribute, and the other for numerical attribute. 
     * 
     * For numerical attribute :
     *       simply find the mean of the datasets, and divide them into two subsets, then calculate
     *       the information gain
     * For categorical attribute:
     *       devide the datasets based on each categorical value, then calculate the information gain
     *       
     * Formula for information gain
     *       IG = H(data) - \sum_{i=1}^{d} p(subdata_{i}) *H(subdata_{i})
     *       
     * @param entropy           entropy for the total data sets. 
     * @param currFeatureIdx    the index of feature attribute we are currently considering
     * @param datasets          the dataset (excluding labels)  
     * @param labels            corresponding labels for these datasets. 
     * @return                  information gain. 
     */
    private double getInformationGain(double entropy, int featureIndex, DataSet dataset){
        int dataSize = dataset.getNumOfInstance();
        // get the mean
        double mean = 0;
        for (int i = 0; i < dataset.getNumOfInstance(); i++){
            double[] featureVector = dataset.getInstance(i).getFeatureVector();
            mean += featureVector[featureIndex]/dataSize;
        }
        
        // divide the dataset into two subset, based on the mean value. 
        int leftSize = 0;
        for (int i = 0; i < dataSize; i++){
            if ((dataset.getInstance(i).getFeatureVector())[featureIndex] < mean)
                leftSize++;
        }
        int rightSize = dataSize - leftSize;
        
        List<Integer> leftLabels = new ArrayList<Integer>(leftSize);
        List<Integer> rightLabels = new ArrayList<Integer>(rightSize);
        
        for (int i = 0; i < dataSize; i++){
            if (dataset.getInstance(i).getFeatureVector()[featureIndex] < mean)
                leftLabels.add(dataset.getInstance(i).getLabelIndex());
            else
                rightLabels.add(dataset.getInstance(i).getLabelIndex());
        }
        
        double leftEntropy = EntropyUtils.getEntropy(leftLabels);
        double rightEntropy = EntropyUtils.getEntropy(rightLabels);
        
        return entropy - (leftSize*1.0/dataSize) * leftEntropy 
                - (rightSize*1.0/dataSize) * rightEntropy;
    } 
            
    /**
     * Create a new decision node. 
     */
    private TreeNode createDecisionNode(int bestFeatureIdx, DataSet dataset){
        // calculate the mean. 
        double mean = 0;
        int dataSize = dataset.getNumOfInstance();
        for (int i = 0; i < dataSize; i++){
            double[] featureVector = dataset.getInstance(i).getFeatureVector();
            mean += featureVector[bestFeatureIdx]/dataSize;
        }
        
        List<Instance> leftDataSet = new ArrayList<Instance>();
        List<Instance> rightDataSet = new ArrayList<Instance>();
        // divide the datasets into two subset, based on the mean value. 
        for (int i = 0; i < dataSize; i++){
            // smaller one goes to left. 
            if ((dataset.getInstance(i).getFeatureVector())[bestFeatureIdx] < mean)
                leftDataSet.add(dataset.getInstance(i));
            else
                rightDataSet.add(dataset.getInstance(i));
        }
          
        // create new decision node, and set the left child and right child. 
        TreeNode node = new DecisionNode(bestFeatureIdx, mean);
        if (leftDataSet.size() > 0){
            ((DecisionNode)node).setLeftChild(build(new DataSet(leftDataSet)));
        } else{
            // create leaf node, with majority distribution. 
            TreeNode leafNode = new LeafNode(numLabels, dataset.getLabels());
            ((DecisionNode)node).setLeftChild(leafNode);
        }
            
        if (rightDataSet.size() > 0){
            ((DecisionNode)node).setRightChild(build(new DataSet(rightDataSet)));
        } else{
            // create leaf node, with majority distribution. 
            TreeNode leafNode = new LeafNode(numLabels, dataset.getLabels());
            ((DecisionNode)node).setRightChild(leafNode);
        }

        return node;
    }
   
    private boolean hasSameLabel(List<Integer> labels){
        for (int i = 1; i < labels.size(); i++){
            if (labels.get(i) != labels.get(i-1))
                return false;
        }
        return true;
    }
   
    public void setSampleFeatureSize(int sampleFeatureSize){
        this.sampleFeatureSize = sampleFeatureSize;
    }
    
    public void setTreeMinSize(int minTreeSize){
        this.minTreeSize = minTreeSize;
    }    
}
