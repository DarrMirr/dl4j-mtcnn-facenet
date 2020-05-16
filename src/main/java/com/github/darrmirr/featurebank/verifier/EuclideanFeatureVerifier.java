package com.github.darrmirr.featurebank.verifier;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component
@Qualifier(FeatureVerifier.EUCLIDEAN_DISTANCE)
public class EuclideanFeatureVerifier implements FeatureVerifier {
    private static final Logger logger = LoggerFactory.getLogger(EuclideanFeatureVerifier.class);

    @Override
    public double verify(INDArray featureSource, INDArray featureTest) {
        var distance = Transforms.euclideanDistance(featureSource, featureTest);
        logger.debug("Euclidean distance : {}", distance);
        return distance;
    }

    @Override
    public double threshold() {
        return 1.1;
    }

    private double euclideanDistance(INDArray featureSource, INDArray featureTest) {
        INDArray tmp = featureSource.sub(featureTest);
        tmp = tmp.mul(tmp).sum(1);
        return tmp.getDouble(0);
    }
}
