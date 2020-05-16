package com.github.darrmirr.featurebank;

import com.github.darrmirr.featurebank.verifier.FeatureVerifier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Feature bank stores features in HashMap structure
 * HashMap could store only one feature vector for the same label (one (feature) to one (label) relation)
 */

@Component
@Qualifier(FeatureBank.HASH_MAP)
public class HashMapFeatureBank implements FeatureBank {
    private static final Logger logger = LoggerFactory.getLogger(HashMapFeatureBank.class);
    private final Map<String, INDArray> featureBank = new HashMap<>();
    private FeatureVerifier featureVerifier;

    @Autowired
    public HashMapFeatureBank(@Qualifier(FeatureVerifier.COSINE_DISTANCE) FeatureVerifier featureVerifier) {
        this.featureVerifier = featureVerifier;
    }

    @Override
    public void put(String label, INDArray featureVector) {
        logger.info("try to add {} to feature bank", label);
        if (label != null && featureVector != null) {
            featureBank.put(label, featureVector);
            logger.info("{} has added to feature bank", label);
        }
    }

    @Override
    public INDArray get(String label) {
        return Optional
                .ofNullable(label)
                .map(featureBank::get)
                .orElse(null);
    }

    @Override
    public INDArray getSimilar(INDArray featureTest) {
        double minVal = Double.MAX_VALUE;
        String label = "none";
        for(Map.Entry<String, INDArray> entry : featureBank.entrySet()) {
            double tmpVal = featureVerifier.verify(entry.getValue(), featureTest);
            logger.info("similarity with {} is {}", entry.getKey(), tmpVal);
            if(tmpVal < minVal) {
                minVal = tmpVal;
                label = entry.getKey();
            }
        }

        if(minVal < featureVerifier.threshold()) {
            logger.info("similarity with {} is {} (min distance)", label, minVal);
            return get(label);
        }
        logger.info("cannot recognize this person, but the similar one is {} ({})", label, minVal);
        return Nd4j.empty();
    }
}
