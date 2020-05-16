package com.github.darrmirr.models.custom;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.PReLUParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

public class PReLUNormParamInitializer extends PReLUParamInitializer {

	public PReLUNormParamInitializer() {
		super(null, null);
	}
	
    @Override
    public long numParams(Layer l) {
        PReLUNorm prelu = (PReLUNorm) l;
        return prelu.getNOut();
    }
    
    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        long length = numParams(conf);
        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length));
        Map<String, INDArray> out = new LinkedHashMap<String, INDArray>();
        out.put(WEIGHT_KEY, weightGradientView);
        return out;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView,
                                          boolean initializeParameters) {

        FeedForwardLayer layerConf = (FeedForwardLayer) conf.getLayer();
        if (initializeParameters) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            return WeightInitUtil.initWeights(layerConf.getNIn(), layerConf.getNOut(),
                    new long[]{1, numParams(conf)}, layerConf.getWeightInit(), dist, weightParamView);
        } else {
            return WeightInitUtil.reshapeWeights(new long[]{1, numParams(conf)}, weightParamView);
        }
    }


}
