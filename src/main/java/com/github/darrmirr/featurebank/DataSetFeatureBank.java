package com.github.darrmirr.featurebank;

import com.github.darrmirr.featurebank.verifier.FeatureVerifier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * Feature bank stores features in ND4J DataSet structure
 * DataSet could store multiple feature vectors for the same label (many (features) to one (label) relation)
 */

@Component
@Qualifier(FeatureBank.DATA_SET)
public class DataSetFeatureBank implements FeatureBank {
    private static final Logger logger = LoggerFactory.getLogger(DataSetFeatureBank.class);
    private DataSet dataSet;
    private FeatureVerifier featureVerifier;

    @Autowired
    public DataSetFeatureBank(@Qualifier(FeatureVerifier.EUCLIDEAN_DISTANCE) FeatureVerifier featureVerifier) {
        this.featureVerifier = featureVerifier;
    }

    @Override
    public void put(String label, INDArray featureVector) {
        logger.info("try to add {} to feature bank", label);
        if (label != null && featureVector != null) {
            if (dataSet == null) {
                dataSet = new DataSet(featureVector, Nd4j.zeros(1));
                dataSet.getLabelNamesList().add(0, label);
            } else {
                int labelIdx = getIndexOf(label);

                dataSet.setFeatures(Nd4j.vstack(dataSet.getFeatures(), featureVector));
                dataSet.setLabels(Nd4j.hstack(dataSet.getLabels(), Nd4j.zeros(1).addi(labelIdx)));
                dataSet.getLabelNamesList().add(labelIdx, label);
            }
            logger.info("{} has added to feature bank", label);
        }
    }

    @Override
    public INDArray get(String label) {
        var labelIdx = dataSet == null ? -1 : dataSet.getLabelNamesList().indexOf(label);
        return labelIdx == -1 ? Nd4j.empty() : dataSet.getFeatures().getRow(labelIdx);
    }

    @Override
    public INDArray getSimilar(INDArray featureTest) {
        double minVal = Double.MAX_VALUE;
        String label = "none";
        for(int i = 0; i < dataSet.numOutcomes(); i++) {
            var featureSource = dataSet.getFeatures().getRow(i);
            double tmpVal = featureVerifier.verify(featureSource, featureTest);
            logger.debug("similarity with {} is {}", dataSet.getLabelName(i), tmpVal);
            if(tmpVal < minVal) {
                minVal = tmpVal;
                label = dataSet.getLabelName(i);
            }
        }

        if(minVal < featureVerifier.threshold()) {
            logger.info("similarity with {} is {} (min distance)", label, minVal);
            return get(label);
        }
        logger.info("cannot recognize this person, but the similar one is {} ({})", label, minVal);
        return Nd4j.empty();
    }

    private int getIndexOf(String label) {
        return dataSet == null ? 0 : dataSet
                .getLabelNamesList()
                .stream()
                .filter(storedLabel ->
                        storedLabel.equalsIgnoreCase(label))
                .findFirst()
                .map(dataSet.getLabelNamesList()::indexOf)
                .orElse(dataSet.getLabelNamesList().size());
    }
}
