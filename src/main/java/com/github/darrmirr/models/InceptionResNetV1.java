package com.github.darrmirr.models;

import com.github.darrmirr.models.custom.ActivationLinear;
import com.github.darrmirr.models.custom.ActivationReverse;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static java.util.Collections.enumeration;
import static org.deeplearning4j.nn.conf.ConvolutionMode.Truncate;

@Component
public class InceptionResNetV1 implements Dl4jModel {
    private static final Logger logger = LoggerFactory.getLogger(InceptionResNetV1.class);
    private static final String WEIGHTS_PATH = "models/inceptionResNetV1/InceptionResNetV1Data";
    private ComputationGraphConfiguration graphConfiguration;

    public InceptionResNetV1(long[] inputShape) {
        try {
            graphConfiguration = buildConfiguration(inputShape);
        } catch (Exception e) {
            logger.error("error to build ComputationGraphConfiguration", e);
        }
    }

    @PostConstruct
    public void init() throws IOException {
        var weightsResource = new ClassPathResource(WEIGHTS_PATH);
        if (!weightsResource.exists()) {
            logger.info("merge weights into one file");
            var weightsDirPath = Paths.get(WEIGHTS_PATH).getParent().toString();
            var weightsDirResource = new ClassPathResource(weightsDirPath);

            var splittedFilesSet = Optional
                    .of(weightsDirResource.getFile())
                    .map(File::listFiles)
                    .map(Arrays::asList)
                    .map(TreeSet::new)
                    .orElseThrow();

            var splittedInputStreams = new LinkedList<InputStream>();
            for (File file : splittedFilesSet) {
                splittedInputStreams.add(new FileInputStream(file));
            }

            try(var inputStream = new SequenceInputStream(enumeration(splittedInputStreams))) {
                var weightsPath = Paths.get(weightsDirResource.getFile().getAbsolutePath(), weightsResource.getFilename());
                Files.copy(inputStream, weightsPath);
            }
        } else {
            logger.info("merge weights into one file is not required");
        }
    }

    public InceptionResNetV1()  {
        this(new long[] { 160, 160, 3 });
    }

    @Override
    public ComputationGraphConfiguration getConfiguration() {
        return graphConfiguration;
    }

    @Override
    public String getWeightsPath() {
        return WEIGHTS_PATH;
    }

    private ComputationGraphConfiguration buildConfiguration(long[] inputShape) throws Exception {
        String input = "input";
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(input)
                .setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]));
        GraphBuilderHelper helper = new GraphBuilderHelper(builder);
        helper.lastLayer = input;
        helper.layerConfMap.put(input, new LayerConf(input, (int) inputShape[2]));
        helper.addLayerAndBatchNormBehind("Conv2d_1a_3x3", defConv(32, 3, 2).convolutionMode(Truncate));
        helper.addLayerAndBatchNormBehind("Conv2d_2a_3x3", defConv(32, 3).convolutionMode(Truncate));
        helper.addLayerAndBatchNormBehind("Conv2d_2b_3x3", defConv(64, 3));
        helper.addLayerBehind("MaxPool_3a_3x3", defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate));
        helper.addLayerAndBatchNormBehind("Conv2d_3b_1x1", defConv(80, 1).convolutionMode(Truncate));
        helper.addLayerAndBatchNormBehind("Conv2d_4a_3x3", defConv(192, 3).convolutionMode(Truncate));
        helper.addLayerAndBatchNormBehind("Conv2d_4b_3x3", defConv(256, 3, 2).convolutionMode(Truncate));
        for (int i = 0; i < 5; i++) {
            block35(helper, "block35_" + i + "/", 0.17);
        }
        reduction_a(helper, 192, 192, 256, 384);
        for (int i = 0; i < 10; i++) {
            block17(helper, "block17_" + i + "/", 0.1);
        }
        reduction_b(helper);
        for (int i = 0; i < 5; i++) {
            block8(helper, true, "block8_" + i, 0.2);
        }
        block8(helper, false, "block8_final", 1);
        helper.addLayerBehind("avg_pool", defPool(PoolingType.AVG, 3).convolutionMode(Truncate));
        helper.addLayerBehind("Dropout", new DropoutLayer.Builder(0.8));
        helper.addLayerBehind("reverse", new ActivationLayer.Builder(new ActivationReverse()));
        helper.addLayerAndBatchNormBehind("Bottleneck", defDense(128), Activation.IDENTITY);
        helper.addLayerBehind("logits", defDense(44052).hasBias(true));
        helper.addVertex("embeddings", new L2NormalizeVertex(new int[] { 1 }, 1e-10), 0, toActName("Bottleneck"));
        builder.setOutputs("logits", "embeddings");
        return builder.build();
    }

    public static INDArray prewhiten(INDArray x){
        double mean = x.mean().getDouble(0);
        double std = x.std().getDouble(0);
        double stdAdj = Math.max(std, 1.0/Math.sqrt(x.length()));
        return x.sub(mean).mul(1/stdAdj);
    }

    private DenseLayer.Builder defDense(int output) {
        double weight_decay = 0;
        return new DenseLayer.Builder().weightInit(WeightInit.DISTRIBUTION)
                .dist(new TruncatedNormalDistribution(0, 0.1)).l2(weight_decay).nOut(output)
                .activation(Activation.IDENTITY).hasBias(false);
    }

    private ConvolutionLayer.Builder defConv(int output, int kernelSize, int stride) {
        double weight_decay = 0;
        return new ConvolutionLayer.Builder(kernelSize, kernelSize).weightInit(WeightInit.DISTRIBUTION)
                .dist(new TruncatedNormalDistribution(0, 0.1)).l2(weight_decay).stride(stride, stride).nOut(output)
                .convolutionMode(ConvolutionMode.Same).activation(Activation.IDENTITY).hasBias(false);
    }

    private ConvolutionLayer.Builder defConv(int output, int kernelSize) {
        return defConv(output, kernelSize, 1);
    }

    private BatchNormalization.Builder defBatchNorm() {
        return new BatchNormalization.Builder(false).decay(0.995).eps(0.001);
    }

    private SubsamplingLayer.Builder defPool(PoolingType type, int kernelSize, int stride) {
        return new SubsamplingLayer.Builder(type).kernelSize(kernelSize, kernelSize).stride(stride, stride);
    }

    private SubsamplingLayer.Builder defPool(PoolingType type, int kernelSize) {
        return defPool(type, kernelSize, 1);
    }

    private String toBatchNormName(String name) {
        return name + "/batch_norm";
    }

    private String toActName(String name) {
        return name + "/act";
    }

    private static String nameLayer(String original, String i) {
        return String.format("(%s)%s", i, original);
    }

    private void block35(GraphBuilderHelper helper, String prefix, final double scale) throws Exception {
        String input = helper.lastLayer;
        String tower_conv = prefix + "/Branch_0/Conv2d_1x1";
        String tower_conv1_0 = prefix + "/Branch_1/Conv2d_0a_1x1";
        String tower_conv1_1 = prefix + "/Branch_1/Conv2d_0b_3x3";
        String tower_conv2_0 = prefix + "/Branch_2/Conv2d_0a_1x1";
        String tower_conv2_1 = prefix + "/Branch_2/Conv2d_0b_3x3";
        String tower_conv2_2 = prefix + "/Branch_2/Conv2d_0c_3x3";
        String mixed = prefix + "/mixed";
        String up = prefix + "/Conv2d_1x1";
        String scaleName = prefix + "/scale";
        String add = prefix + "/add";
        String relu = prefix + "/relu";
        helper.addLayerAndBatchNorm(tower_conv, defConv(32, 1), input);
        helper.addLayerAndBatchNorm(tower_conv1_0, defConv(32, 1), input);
        helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(32, 3));
        helper.addLayerAndBatchNorm(tower_conv2_0, defConv(32, 1), input);
        helper.addLayerAndBatchNormBehind(tower_conv2_1, defConv(32, 3));
        helper.addLayerAndBatchNormBehind(tower_conv2_2, defConv(32, 3));
        String[] merges = new String[] { toActName(tower_conv2_2), toActName(tower_conv1_1), toActName(tower_conv) };
        helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
        helper.addLayer(up, defConv(helper.getOutput(input), 1).hasBias(true), mixed);
        helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
        helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
                scaleName);
        helper.addLayer(relu, new ActivationLayer.Builder().activation(Activation.RELU), add);
    }

    private void block17(GraphBuilderHelper helper, String i, final double scale) throws Exception {
        String b0 = nameLayer("b0", i), b1_0 = nameLayer("b1_0", i), b1_1 = nameLayer("b1_1", i),
                b1_2 = nameLayer("b2_0", i), mixed = nameLayer("mixed", i), up = nameLayer("up", i),
                scaleName = nameLayer("scaleName", i), add = nameLayer("add", i),
                activation = nameLayer("activation", i);
        String input = helper.lastLayer;
        helper.addLayerAndBatchNorm(b0, defConv(128, 1), input);
        helper.addLayerAndBatchNorm(b1_0, defConv(128, 1), input);
        helper.addLayerAndBatchNormBehind(b1_1, defConv(128, 1).kernelSize(1, 7));
        helper.addLayerAndBatchNormBehind(b1_2, defConv(128, 1).kernelSize(7, 1));
        String[] merges = new String[] { toActName(b1_2), toActName(b0) };
        helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
        helper.addLayer(up, defConv(helper.getOutput(input), 1).hasBias(true), mixed);
        helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
        helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
                scaleName);
        helper.addLayer(activation, new ActivationLayer.Builder().activation(Activation.RELU), add);
    }

    private void block8(GraphBuilderHelper helper, boolean activateFunc, String i, final double scale)
            throws Exception {
        String b0 = nameLayer("b0", i), b1_0 = nameLayer("b1_0", i), b1_1 = nameLayer("b1_1", i),
                b1_2 = nameLayer("b2_0", i), mixed = nameLayer("mixed", i), up = nameLayer("up", i),
                scaleName = nameLayer("scaleName", i), add = nameLayer("add", i),
                activation = nameLayer("activation", i);
        String input = helper.lastLayer;
        helper.addLayerAndBatchNormBehind(b0, defConv(192, 1));
        helper.addLayerAndBatchNorm(b1_0, defConv(192, 1), input);
        helper.addLayerAndBatchNormBehind(b1_1, defConv(192, 1).kernelSize(1, 3));
        helper.addLayerAndBatchNormBehind(b1_2, defConv(192, 1).kernelSize(3, 1));
        String[] merges = new String[] { toActName(b1_2), toActName(b0) };
        helper.addVertex(mixed, new MergeVertex(), helper.getOutput(merges), merges);
        helper.addLayerBehind(up, defConv(helper.getOutput(input), 1).hasBias(true));
        helper.addLayerBehind(scaleName, new ActivationLayer.Builder(new ActivationLinear(scale)));
        helper.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), helper.getOutput(input), input,
                scaleName);
        Activation act = activateFunc ? Activation.RELU : Activation.IDENTITY;
        helper.addLayer(activation, new ActivationLayer.Builder().activation(act), add);
    }

    private void reduction_a(GraphBuilderHelper helper, int k, int l, int m, int n) throws Exception {
        String tower_conv = nameLayer("tower_conv", "a"), tower_conv1_0 = nameLayer("tower_conv1_0", "a"),
                tower_conv1_1 = nameLayer("tower_conv1_1", "a"), tower_conv1_2 = nameLayer("tower_conv1_2", "a"),
                tower_pool = nameLayer("tower_pool", "a"), res = nameLayer("res", "a");
        String input = helper.lastLayer;
        helper.addLayerAndBatchNormBehind(tower_conv, defConv(n, 3, 2).convolutionMode(Truncate));
        helper.addLayerAndBatchNorm(tower_conv1_0, defConv(k, 1), input);
        helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(l, 3));
        helper.addLayerAndBatchNormBehind(tower_conv1_2, defConv(m, 3, 2).convolutionMode(Truncate));
        helper.addLayer(tower_pool, defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate), input);
        String[] merges = new String[] { tower_pool, toActName(tower_conv1_2), toActName(tower_conv) };
        helper.addVertex(res, new MergeVertex(), helper.getOutput(merges), merges);
    }

    private void reduction_b(GraphBuilderHelper helper) throws Exception {
        String tower_conv = nameLayer("tower_conv", "b"), tower_conv_1 = nameLayer("tower_conv_1", "b"),
                tower_conv1 = nameLayer("tower_conv1_0", "b"), tower_conv1_1 = nameLayer("tower_conv1_1", "b"),
                tower_conv2 = nameLayer("tower_conv2", "b"), tower_conv2_1 = nameLayer("tower_conv2_1", "b"),
                tower_conv2_2 = nameLayer("tower_conv2_2", "b"), tower_pool = nameLayer("tower_pool", "b"),
                res = nameLayer("res", "b");
        String input = helper.lastLayer;
        helper.addLayerAndBatchNormBehind(tower_conv, defConv(256, 1));
        helper.addLayerAndBatchNormBehind(tower_conv_1, defConv(384, 3, 2).convolutionMode(Truncate));
        helper.addLayerAndBatchNorm(tower_conv1, defConv(256, 1), input);
        helper.addLayerAndBatchNormBehind(tower_conv1_1, defConv(256, 3, 2).convolutionMode(Truncate));
        helper.addLayerAndBatchNorm(tower_conv2, defConv(256, 1), input);
        helper.addLayerAndBatchNormBehind(tower_conv2_1, defConv(256, 3));
        helper.addLayerAndBatchNormBehind(tower_conv2_2, defConv(256, 3, 2).convolutionMode(Truncate));
        helper.addLayer(tower_pool, defPool(PoolingType.MAX, 3, 2).convolutionMode(Truncate), input);
        String[] merges = new String[] { tower_pool, toActName(tower_conv2_2), toActName(tower_conv1_1),
                toActName(tower_conv_1) };
        helper.addVertex(res, new MergeVertex(), helper.getOutput(merges), merges);
    }

    private class GraphBuilderHelper {
        ComputationGraphConfiguration.GraphBuilder builder;
        String lastLayer = null;
        Map<String, LayerConf> layerConfMap = new HashMap<String, LayerConf>();
        Field outField;

        public GraphBuilderHelper(ComputationGraphConfiguration.GraphBuilder builder) throws Exception {
            super();
            this.builder = builder;
            outField = FeedForwardLayer.Builder.class.getDeclaredField("nOut");
            outField.setAccessible(true);
        }

        int getOutput(String layerName) {
            LayerConf conf = layerConfMap.get(layerName);
            if (conf == null) {
                throw new RuntimeException("unknown layer:" + layerName);
            }
            if (conf.nOut > 0) {
                return conf.nOut;
            }
            return getOutput(conf.input);
        }

        int getOutput(String[] layerNames) {
            int sum = 0;
            for (String input : layerNames) {
                sum += getOutput(input);
            }
            return sum;
        }

        void addLayer(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, String... input)
                throws Exception {
            int nOut = 0;
            InputPreProcessor inputPreProcessor = null;
            if (layer instanceof FeedForwardLayer.Builder) {
                nOut = outField.getInt(layer);
            }
            if (input != null && input.length > 0 && "input".equalsIgnoreCase(input[0])){
                inputPreProcessor = new InceptionResNetInputPreProcessor();
            }

            builder.addLayer(name, layer.build(), inputPreProcessor, input);
            lastLayer = name;
            layerConfMap.put(name, new LayerConf(name, nOut, input));
        }

        void addLayerBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer) throws Exception {
            if (lastLayer == null) {
                throw new RuntimeException("no last layer");
            }
            addLayer(name, layer, lastLayer);
        }

        void addLayerAndBatchNorm(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, Activation act,
                                  String... input) throws Exception {
            addLayer(name, layer, input);
            addLayerBehind(toBatchNormName(name), defBatchNorm());
            addLayerBehind(toActName(name), new ActivationLayer.Builder().activation(act));
        }

        void addLayerAndBatchNorm(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, String... input)
                throws Exception {
            addLayerAndBatchNorm(name, layer, Activation.RELU, input);
        }

        void addLayerAndBatchNormBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer, Activation act)
                throws Exception {
            addLayerAndBatchNorm(name, layer, act, lastLayer);
        }

        void addLayerAndBatchNormBehind(String name, @SuppressWarnings("rawtypes") Layer.Builder layer)
                throws Exception {
            addLayerAndBatchNormBehind(name, layer, Activation.RELU);
        }

        void addVertex(String vertexName, GraphVertex vertex, int nOut, String... vertexInputs) {
            builder.addVertex(vertexName, vertex, vertexInputs);
            lastLayer = vertexName;
            layerConfMap.put(vertexName, new LayerConf(vertexName, nOut, vertexInputs));
        }
    }

    private static class LayerConf {
        int nOut;
        String[] input;

        public LayerConf(String name, int nOut, String... input) {
            super();
            this.nOut = nOut;
            this.input = input;
        }
    }
}
