package com.github.darrmirr.models;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Input preprocessor for input neural network layer
 */
public class InceptionResNetInputPreProcessor extends BaseInputPreProcessor {

    @Override
    public INDArray preProcess(INDArray indArray, int i, LayerWorkspaceMgr layerWorkspaceMgr) {
        double mean = indArray.mean().getDouble(0);
        double std = indArray.std().getDouble(0);
        double stdAdj = Math.max(std, 1.0 / Math.sqrt(indArray.length()));
        return indArray.sub(mean).mul(1 / stdAdj);
    }

    @Override
    public INDArray backprop(INDArray indArray, int i, LayerWorkspaceMgr layerWorkspaceMgr) {
        return indArray;
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        return inputType;
    }
}
