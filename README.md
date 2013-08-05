RandomForest, Java Implementation
=============================

Random forest is a bagging approach for combining multiple decision trees. The prediction is aggregated 
across all of the trees. The code is easy to understand, so good for someone who wants to learn about random forest. 
You can run through the example files to easily figure out how it works. 

The process of building random forest (used in this implementation) as follows:

N: # of instances in training data
M: # of trees in the random forest
D: # of attributes for each instance (size of feature vector)

<ul>
 <li>(a) Draw a bootstrap sample Z* of size N from the training data (with replacement) </li>
 <li>(b) Build a tree T_{i} for the bootstrap data , by recursively
      repeating the following steps for each terminal node of the tree, until 
      the minimum node size MIN_TREE_SIZE is reached. 
    <ul>
      <li>(i) Randomly select m variables from the total feature set (D). </li>
      <li> (ii) Pick the best variable/split-point among the m (using information gain)</li>
      <li> (iii) Split the node into left and right child. </li>
    </ul>
 <\li>
</ul>


The calculation of entropy and information gain, 
refer to : http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/mlbook/ch3.pdf


Code Example (you can also find it in the SimpleExample.java:

```java
// create artificial training data. 
List<double[]> data = new ArrayList<double[]>();
data.add(new double[]{1,2});
data.add(new double[]{2,4});
data.add(new double[]{2,5});
data.add(new double[]{3,4});
data.add(new double[]{5,1});
data.add(new double[]{5,2});
data.add(new double[]{5,3});
data.add(new double[]{5,4});

List<Integer> label = new ArrayList<Integer>();
label.add(0);
label.add(0);
label.add(0);
label.add(0);
       

        
```