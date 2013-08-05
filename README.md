RandomForest, Java Implementation
=============================

Random forest is a bagging approach for combining multiple decision trees. The prediction is aggregated 
across all of the trees. The code is easy to understand, so good for someone who wants to learn about random forest. 
You can run through the example files to easily figure out how it works. 


<ul>
<li>N: # of instances in training data</li>
<li>M: # of trees in the random forest</li>
<li>D: # of attributes for each instance (size of feature vector)</li>
</ul>



The process of building random forest (used in this implementation) as follows:
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


<b>Code Example (you can also find it in the SimpleExample.java:<b>
```java
// create artificial training data for binary classification(also work for multi-class). 
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
label.add(1);
label.add(1);
label.add(1);
label.add(1);   
```

<b> Build random forest <b>
```java
// build random forest. 100: # of trees to create   2: # of random features to select 
// when deciding split the node.
RandomForest rf = new RandomForest(new DataSet(data, label), 100, 2);
```
<b> Test </b>
```java
List<double[]> testData = new ArrayList<double[]>();
testData.add(new double[]{3,5});   
testData.add(new double[]{1,10});  
for (double[] testSample: testData){
    System.out.println(rf.predictLabel(testSample)); // you can also predict label probability. 
}
```
