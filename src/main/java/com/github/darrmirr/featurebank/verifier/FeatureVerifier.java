package com.github.darrmirr.featurebank.verifier;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface FeatureVerifier {
    String COSINE_DISTANCE = "cosine_distance";
    String EUCLIDEAN_DISTANCE = "euclidean_distance";

    double verify(INDArray featureSource, INDArray featureTest);

    double threshold();
}
