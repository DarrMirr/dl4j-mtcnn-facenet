package com.github.darrmirr.models.mtcnn.networks.dl4j;

import com.github.darrmirr.models.Dl4jModel;
import com.github.darrmirr.models.custom.PReLUNorm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;
import org.springframework.stereotype.Component;

@Component
public class RefineNetModel implements Dl4jModel {
    private static final String WEIGHTS_PATH = "models/mtcnn/weights/RNetData";
    private ComputationGraphConfiguration graphConfiguration;

    public RefineNetModel(long[] inputShape) {
        graphConfiguration = buildConfiguration(inputShape);
    }

    public RefineNetModel() {
        this(new long[] { 24, 24, 3 });
    }

    private ComputationGraphConfiguration buildConfiguration(long[] inputShape) {
        String input = "input";
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
                .setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]))
                .addLayer("conv1",new ConvolutionLayer.Builder(3, 3)
                                .nOut(28)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        input)
                .addLayer("prelu1", new PReLUNorm.Builder().build(), "conv1")
                .addLayer("pool1", new SubsamplingLayer.Builder(3,3)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "prelu1")
                .addLayer("conv2",new ConvolutionLayer.Builder(3, 3)
                                .nOut(48)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool1")
                .addLayer("prelu2", new PReLUNorm.Builder().build(), "conv2")
                .addLayer("pool2", new SubsamplingLayer.Builder(3,3)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(), "prelu2")
                .addLayer("conv3",new ConvolutionLayer.Builder(2, 2)
                                .nOut(64)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool2")
                .addLayer("prelu3", new PReLUNorm.Builder().build(), "conv3")
                .addLayer("conv4", new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.IDENTITY).build(), "prelu3")
                .addLayer("prelu4", new PReLUNorm.Builder().build(), "conv4")
                .addLayer("conv5-1",new DenseLayer.Builder()
                                .nOut(2)
                                .activation(Activation.SOFTMAX).build(),
                        "prelu4")
                .addLayer("conv5-2", new DenseLayer.Builder().nOut(4)
                        .activation(Activation.IDENTITY).build(), "prelu4")
                .setOutputs("conv5-1", "conv5-2");
        return builder.build();
    }

    @Override
    public ComputationGraphConfiguration getConfiguration() {
        return graphConfiguration;
    }

    @Override
    public String getWeightsPath() {
        return WEIGHTS_PATH;
    }
}
