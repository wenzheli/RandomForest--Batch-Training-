package com.ml.rf.nodes;

import java.util.List;

/**
 * Leaf node for decision tree. Leaf node either contains several data points with 
 * the same label, or several data points with different labels. The reason that includes
 * multiple labels is that, when we create a new subtree using subset of samples, whose
 * size is less than MIN_SUBTREE_SIZE, we stop splitting them. Instead we just create 
 * a leaf node. 
 * 
 * @author wenzhe
 *
 */
public class LeafNode extends TreeNode{
    
    // probability distributions for each label. 
    private double[] labelProbDist; 
    
    
    /**
     * Create a new leaf node, by taking all response variables. 
     * Basically, it will create a histogram.  For example, if input param
     * are (3, (0,0,1)), it will create a distribution (0.67, 0.33, 0). 
     * 
     * @param numLabels   total number of labels 
     * @param labels   list of response labels
     */
    public LeafNode(int numLabels, List<Integer> labels){
        super();
        labelProbDist = new double[numLabels];
        int size = labels.size();
        for (Integer label : labels){
            labelProbDist[label] += 1.0/size;
        }
    }
   
    public double[] getLabelProbDist(){
        return labelProbDist;
    }   
}
