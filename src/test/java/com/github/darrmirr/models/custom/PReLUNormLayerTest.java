package com.github.darrmirr.models.custom;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.mockito.Mockito;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class PReLUNormLayerTest {
    private NeuralNetConfiguration mockNeuralNetConfiguration = Mockito.mock(NeuralNetConfiguration.class);
    private PReLUNormLayer layer = new PReLUNormLayer(mockNeuralNetConfiguration);

    @Test
    public void activate01() throws IOException {
        ClassPathResource input = new ClassPathResource("layers/activate/01-input.ind");
        ClassPathResource output = new ClassPathResource("layers/activate/01-output.ind");
        ClassPathResource alpha = new ClassPathResource("layers/activate/01-alpha.ind");

        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());
        var alphaArray = Nd4j.readBinary(alpha.getFile());

        var activation = new ActivationPReLU(alphaArray, null);
        var actualArray = layer.activate(inputArray, activation, false);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }

    @Test
    public void activate03() throws IOException {
        ClassPathResource input = new ClassPathResource("layers/activate/03-input.ind");
        ClassPathResource output = new ClassPathResource("layers/activate/03-output.ind");
        ClassPathResource alpha = new ClassPathResource("layers/activate/03-alpha.ind");

        var inputArray = Nd4j.readBinary(input.getFile());
        var outputArray = Nd4j.readBinary(output.getFile());
        var alphaArray = Nd4j.readBinary(alpha.getFile());

        var activation = new ActivationPReLU(alphaArray, null);
        var actualArray = layer.activate(inputArray, activation, false);

        assertThat(actualArray.eq(outputArray).minNumber(), is(1.0));
    }
}
