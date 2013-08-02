package com.ml.rf.utils;

import java.util.Random;

public class SamplerUtils {
    /**
     * Random sample m elements from [0,1,...,n-1].
     * TODO more efficient sampler, based on Kuth's algorithm.  
     * Example:
     *      randSample(10, 3) will return [3,6,7]
     * @param n     total number of elements.
     * @param m     the sample size
     * @return       
     */
    public static int[] randSample(int n, int m)
    {   
        int[] indexs = new int[n];
        for (int i = 0; i < n; i++)
            indexs[i] = i;
        
        for (int i = 0; i < m; i++){
            int randomNum = new Random().nextInt(n - i ) + i;
            // swatch i and randomNum
            int tmp = indexs[i];
            indexs[i] = indexs[randomNum];
            indexs[randomNum] = tmp;
        }
        
        int[] result = new int[m];
        for (int i = 0; i < m; i++)
            result[i] = indexs[i];
        
        return result;
    }
    
    /**
     * bootstrap index of samples, with replacement. 
     *
     */
    public static int[] bootStrap(int n){
        int[] bootstrapIndex = new int[n];
        for (int i = 0; i < n; i++){
            bootstrapIndex[i] = new Random().nextInt(n); 
        }
        
        return bootstrapIndex;
    }
}
