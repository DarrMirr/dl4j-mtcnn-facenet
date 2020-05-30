package com.github.darrmirr.models;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.io.InputStream;

public interface Dl4jModel {

    ComputationGraphConfiguration getConfiguration();

    String getWeightsPath();

    int inputWidth();

    int inputHeight();

    default ComputationGraph getGraph() throws IOException {
        ComputationGraph graph = new ComputationGraph(getConfiguration());
        graph.init();
        var weightsResource = new ClassPathResource(getWeightsPath());
        loadWeightData(weightsResource, graph);
        return graph;
    }

    default void loadWeightData(Resource weightsResource, ComputationGraph graph) throws IOException {
        try (InputStream in = weightsResource.getInputStream()) {
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
