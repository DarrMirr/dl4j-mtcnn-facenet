package com.github.darrmirr.models.custom;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class PReLUNorm extends FeedForwardLayer {

	private static final long serialVersionUID = 1L;

	private PReLUNorm(Builder builder) {
		super(builder);
		initializeConstraints(builder);
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {
		if (inputType == null)
			throw new IllegalStateException("Invalid input type: null for layer name \"" + getLayerName() + "\"");
		return inputType;
	}

	@Override
	public void setNIn(InputType inputType, boolean override) {
		if (nIn <= 0 || override) {
			switch (inputType.getType()) {
			case FF:
				nIn = ((InputType.InputTypeFeedForward) inputType).getSize();
				break;
			case CNN:
				nIn = ((InputType.InputTypeConvolutional) inputType).getChannels();
				break;
			case CNNFlat:
				nIn = ((InputType.InputTypeConvolutionalFlat) inputType).getDepth();
			default:
				throw new IllegalStateException(
						"Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got "
								+ inputType + " for layer " + getLayerName() + "\"");
			}
			nOut = nIn;
		}
	}

	@Override
	public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
		if (inputType.getType() == InputType.Type.CNN) {
			return null;
		}
		return super.getPreProcessorForInputType(inputType);
	}

	// for beta-2 version
	@Override
	public Layer instantiate(
			NeuralNetConfiguration conf,
			Collection<TrainingListener> trainingListeners,
			int layerIndex,
			INDArray layerParamsView,
			boolean initializeParams
	) {
		PReLUNormLayer ret = new PReLUNormLayer(conf);
		ret.setListeners(trainingListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return new PReLUNormParamInitializer();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		InputType outputType = getOutputType(-1, inputType);

		long numParams = initializer().numParams(this);
		long updaterStateSize = (int) getIUpdater().stateSize(numParams);

		return new LayerMemoryReport.Builder(layerName, PReLUNorm.class, inputType, outputType)
				.standardMemory(numParams, updaterStateSize).workingMemory(0, 0, 0, 0)
				.cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS).build();
	}

	public static class Builder extends FeedForwardLayer.Builder<PReLUNorm.Builder> {

		@Override
		@SuppressWarnings("unchecked")
		public PReLUNorm build() {
			return new PReLUNorm(this);
		}
	}

}
