package com.github.darrmirr.models;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;
import java.util.function.Supplier;

public interface Dl4jModel {

    ComputationGraphConfiguration getConfiguration();

    Supplier<InputStream> modelWeights();

    int inputWidth();

    int inputHeight();

    default ComputationGraph getGraph() throws IOException {
        ComputationGraph graph = new ComputationGraph(getConfiguration());
        graph.init();
        loadWeightsTo(graph);
        return graph;
    }

    default void loadWeightsTo(ComputationGraph graph) throws IOException {
        try (InputStream in = modelWeights().get()) {
            var buf = new byte[4];
            var layers = graph.getLayers();
            for (org.deeplearning4j.nn.api.Layer l : layers) {
                int nParams = (int) l.numParams();
                if (nParams == 0)
                    continue;
                float[] data = new float[nParams];
                for (int i = 0; i < nParams; i++) {
                    in.read(buf);
                    data[i] = bytes2Float(buf);
                }
                l.setParams(Nd4j.create(data));
            }
        }
    }

    default float bytes2Float(byte[] arr) {
        int value = 0;
        for (int i = 0; i < 4; i++) {
            value |= ((int) (arr[i] & 0xff)) << (8 * i);
        }
        return Float.intBitsToFloat(value);
    }
}
