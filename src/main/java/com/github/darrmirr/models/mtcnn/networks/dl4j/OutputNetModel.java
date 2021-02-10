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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.function.Supplier;

@Component
public class OutputNetModel implements Dl4jModel {
    private static final Logger logger = LoggerFactory.getLogger(OutputNetModel.class);
    private static final String WEIGHTS_PATH = "models/mtcnn/weights/ONetData";
    private ComputationGraphConfiguration graphConfiguration;

    public OutputNetModel(long[] inputShape) {
        graphConfiguration = buildConfiguration(inputShape);
    }

    public OutputNetModel() {
        this(new long[] { 48, 48, 3 });
    }

    @Override
    public int inputWidth() {
        return 48;
    }

    @Override
    public int inputHeight() {
        return 48;
    }

    private ComputationGraphConfiguration buildConfiguration(long[] inputShape) {
        String input = "input";
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
                .setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]))
                .addLayer("conv1",new ConvolutionLayer.Builder(3, 3)
                                .nOut(32)
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
                                .nOut(64)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool1")
                .addLayer("prelu2", new PReLUNorm.Builder().build(), "conv2")
                .addLayer("pool2", new SubsamplingLayer.Builder(3,3)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(), "prelu2")
                .addLayer("conv3",new ConvolutionLayer.Builder(3, 3)
                                .nOut(64)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool2")
                .addLayer("prelu3", new PReLUNorm.Builder().build(), "conv3")
                .addLayer("pool3", new SubsamplingLayer.Builder(2,2)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "prelu3")
                .addLayer("conv4",new ConvolutionLayer.Builder(2,2)
                                .nOut(128)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool3")
                .addLayer("prelu4", new PReLUNorm.Builder().build(), "conv4")
                .addLayer("conv5", new DenseLayer.Builder().nOut(256)
                        .activation(Activation.IDENTITY).build(), "prelu4")
                .addLayer("prelu5", new PReLUNorm.Builder().build(), "conv5")
                .addLayer("conv6-1", new DenseLayer.Builder().nOut(2)
                        .activation(Activation.SOFTMAX).build(), "prelu5")
                .addLayer("conv6-2", new DenseLayer.Builder().nOut(4)
                        .activation(Activation.IDENTITY).build(), "prelu5")
                .addLayer("conv6-3", new DenseLayer.Builder().nOut(10)
                        .activation(Activation.IDENTITY).build(), "prelu5")
                .setOutputs("conv6-1", "conv6-2", "conv6-3");
        return builder.build();
    }

    @Override
    public ComputationGraphConfiguration getConfiguration() {
        return graphConfiguration;
    }

    @Override
    public Supplier<InputStream> modelWeights() {
        return () -> {
            try {
                var resource = new ClassPathResource(WEIGHTS_PATH);
                try(InputStream is = resource.getInputStream()) {
                    return new ByteArrayInputStream(is.readAllBytes());
                }
            } catch (IOException e) {
                logger.error("error to get model weights", e);
                throw new IllegalStateException(e);
            }
        };
    }
}
