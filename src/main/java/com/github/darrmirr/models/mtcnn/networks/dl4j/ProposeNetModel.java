package com.github.darrmirr.models.mtcnn.networks.dl4j;

import com.github.darrmirr.models.Dl4jModel;
import com.github.darrmirr.models.custom.ActivationSoftMaxAxis;
import com.github.darrmirr.models.custom.PReLUNorm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
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
public class ProposeNetModel implements Dl4jModel {
    private static final Logger logger = LoggerFactory.getLogger(ProposeNetModel.class);
    private static final String WEIGHTS_PATH = "models/mtcnn/weights/PNetData";
    private ComputationGraphConfiguration graphConfiguration;

    public ProposeNetModel(long channels){
        this.graphConfiguration = buildConfiguration(channels);
    }

    public ProposeNetModel() {
        this(3);
    }

    @Override
    public int inputWidth() {
        return -1;
    }

    @Override
    public int inputHeight() {
        return -1;
    }

    private ComputationGraphConfiguration buildConfiguration(long channels) {
        String input = "input";
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
                .addLayer("conv1",new ConvolutionLayer.Builder(3, 3)
                                .nIn(channels)
                                .nOut(10)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        input)
                .addLayer("prelu1", new PReLUNorm.Builder()
                        .nIn(10)
                        .nOut(10).build(), "conv1")
                .addLayer("pool1", new SubsamplingLayer.Builder(2,2)
                        .stride(2,2)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "prelu1")
                .addLayer("conv2",new ConvolutionLayer.Builder(3, 3)
                                .nIn(10)
                                .nOut(16)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "pool1")
                .addLayer("prelu2", new PReLUNorm.Builder()
                        .nIn(16)
                        .nOut(16)
                        .build(), "conv2")
                .addLayer("conv3",new ConvolutionLayer.Builder(3, 3)
                                .nIn(16)
                                .nOut(32)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .activation(Activation.IDENTITY).build(),
                        "prelu2")
                .addLayer("prelu3", new PReLUNorm.Builder()
                        .nIn(32)
                        .nOut(32).build(), "conv3")
                .addLayer("conv4-1",new ConvolutionLayer.Builder(1,1)
                                .nIn(32)
                                .nOut(2)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .activation(new ActivationSoftMaxAxis(1)).build(),
                        "prelu3")
                .addLayer("conv4-2",new ConvolutionLayer.Builder(1,1)
                                .nIn(32)
                                .nOut(4)
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .activation(Activation.IDENTITY).build(),
                        "prelu3")
                .setOutputs("conv4-1","conv4-2");
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
