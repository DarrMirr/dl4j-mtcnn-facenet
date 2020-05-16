package com.github.darrmirr.featurebank;

import org.nd4j.linalg.api.ndarray.INDArray;

/*
 * Interface for features vector storage
 */
public interface FeatureBank {
    String HASH_MAP = "hash_map";
    String DATA_SET = "data_set";

    /**
     * Put new feature vector to bank
     *
     * @param label label for input feature
     * @param featureVector feature vector to store at bank
     */
    void put(String label, INDArray featureVector);

    /**
     * Get stored feature vector by label
     *
     * @param label label name
     * @return feature vector stored at bank
     */
    INDArray get(String label);

    /**
     * Get stored feature vector similar to input one
     *
     * @param featureTest input feature vector to test
     * @return feature vector stored at bank similar to input one
     */
    INDArray getSimilar(INDArray featureTest);
}
