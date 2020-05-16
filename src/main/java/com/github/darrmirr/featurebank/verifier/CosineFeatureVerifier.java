package com.github.darrmirr.featurebank.verifier;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component
@Qualifier(FeatureVerifier.COSINE_DISTANCE)
public class CosineFeatureVerifier implements FeatureVerifier {
    private static final Logger logger = LoggerFactory.getLogger(CosineFeatureVerifier.class);

    @Override
    public double verify(INDArray featureSource, INDArray featureTest) {
        var distance = Transforms.cosineDistance(featureSource, featureTest);
        logger.debug("Cosine distance : {}", distance);
        return distance;
    }

    @Override
    public double threshold() {
        return 0.4;
    }
}
