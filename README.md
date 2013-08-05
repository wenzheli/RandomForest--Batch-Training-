RandomForest, Java Implementation
=============================

Random forest is a bagging approach for combining multiple decision trees. The prediction is aggregated 
across all of the trees. 

The process of building random forest is :
<ul>
 <li>(a) Draw a bootstrap sample Z* of size N from the training data </li>
 <li> (b) Grow a random-forest tree T_{b} to the bootstrap data , by recursively
      repeating the following steps for each terminal node of the tree, until 
      the minimum node size MIN_TREE_SIZE is reached. </li>

       <li>(i) Select m variables at random from the total feature set. </li>
      <li> (ii) Pick the best variable/split-point among the m </li>
      <li> (iii) Split the node into left and right child. </li>

 </li>
</ul>

