package com.github.darrmirr.models.mtcnn.networks;

import com.github.darrmirr.models.mtcnn.MtcnnUtils;
import com.github.darrmirr.models.mtcnn.networks.dl4j.ProposeNetModel;
import com.github.darrmirr.utils.Nd4jUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Component
public class ProposeNet {
    private static final Logger logger = LoggerFactory.getLogger(ProposeNet.class);
    private ProposeNetModel proposeNetModel;
    private MtcnnUtils mtcnnUtils;
    private ComputationGraph graph;
    private int minSize = 20;
    private double factor = 0.709;

    @Autowired
    public ProposeNet(ProposeNetModel proposeNetModel, MtcnnUtils mtcnnUtils) {
        this.proposeNetModel = proposeNetModel;
        this.mtcnnUtils = mtcnnUtils;
    }

    @PostConstruct
    public void init() throws IOException {
        graph = proposeNetModel.getGraph();
    }

    /**
     * Stage 1:
     * 01. Pass in image (method parameters)
     * 02. Calculate multiple scaled factors in order to create scaled copies of the image
     * 03. Feed scaled images into P-Net
     * 04. Gather P-Net output
     * 05. Find indices of bounding boxes with high confidence
     * 06. Convert 12 x 12 kernel coordinates to “un-scaled image” coordinates
     * 07. Non-Maximum Suppression for kernels in each scaled image
     * 08. Non-Maximum Suppression for all kernels
     * 09. Convert bounding box coordinates to “un-scaled image” coordinates
     * 10. Reshape bounding boxes to square
     *
     * @param img input image to find proposal faces
     * @param threshold threshold for bounding box with low confidence
     * @return array of bounding box face candidates
     */
    public INDArray execute(INDArray img, double threshold) {
        logger.debug("propose net : started.");
        // 02. Calculate multiple scaled factors in order to create scaled copies of the image
        double scales[] = getScales(img, minSize, factor);
        logger.debug("scales loaded : {}.", Arrays.toString(scales));
        INDArray totalBoxes = null;

        for (double scale : scales) {
            // 03.- 06.
            var boxes = evaluateBoxes(img, threshold, scale);
            if (boxes == null) {
                continue;
            }
            // 07. Non-Maximum Suppression for kernels in each scaled image
            boxes = mtcnnUtils.nms(boxes, 0.5, false);
            totalBoxes = mergeBoxes(totalBoxes, boxes);
        }

        if (totalBoxes == null) {
            return null;
        }
        // 08. Non-Maximum Suppression for all kernels
        totalBoxes = mtcnnUtils.nms(totalBoxes, 0.7, false);
        // 09. Convert bounding box coordinates to “un-scaled image” coordinates
        totalBoxes = mtcnnUtils.bbreg(totalBoxes);
        long[] imgShape = mtcnnUtils.shape(img);
        // 10. Reshape bounding boxes to square
        totalBoxes = mtcnnUtils.rerec(totalBoxes, imgShape[3], imgShape[2]);
        logger.debug("propose net : finished.");
        return totalBoxes;
    }

    /**
     * Compute the image scale pyramid
     * Image scale pyramid is used to detect faces of all different sizes.
     * In other words, we want to create different copies of the same image in different sizes
     * to search for different sized faces within the image.
     *
     * notice: 12 number is kernel size used in Propose network
     *
     * @param img image array
     * @param minSize
     * @param factor base factor used to calculate image scale factor
     * @return array of image scale factors
     *         (image scale factor is number in percents. It points
     *         how resized image is smaller than original one)
     */
    private double[] getScales(INDArray img, int minSize, double factor) {
        ArrayList<Double> scales = new ArrayList<>();
        long[] imgShape = mtcnnUtils.shape(img);
        assert imgShape.length == 4;
        double m = 12.0 / minSize;
        double minLength = Math.min(imgShape[2], imgShape[3]) * m;
        int factorCount = 0;
        while (minLength >= 12) {
            scales.add(m * Math.pow(factor, factorCount));
            minLength *= factor;
            factorCount++;
        }
        double[] ret = new double[scales.size()];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = scales.get(i).doubleValue();
        }
        return ret;
//        return scales.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Evaluate region object proposal and generate bounding boxes
     *
     * notices:
     *    out[0] is cls layer (face classification layer) output.
     *           Contains scores whether there is object or not for founded boxes
     *    out[1] is reg layer (bounding box regression layer) output.
     *           Contains coordinates (box center coordinates, width and height) of founded boxes
     *
     * @param img image array
     * @param threshold threshold used to reduce bounding box amount.
     *                  There is object inside bounced box if score >= threshold
     *                  Threshold is usually equals to 0.7.
     * @param scale image scale factor is used for resize image
     * @return bounding boxes of object is found by propose net
     */
    private INDArray evaluateBoxes(INDArray img,  double threshold, double scale){
        INDArray pnetInput = mtcnnUtils.transposeBorder(scaleAndNorm(img, scale));
        // 03. Feed scaled images into P-Net
        INDArray[] out = graph.output(pnetInput);
        // 04. Gather P-Net output
        INDArray score = out[0].get(point(0), point(0), all(), all());
        INDArray reg = out[1];
        // 05. - 06.
        return mtcnnUtils.generateBox(score, reg, threshold, scale);
    }

    public INDArray scaleAndNorm(INDArray img, double scale) {
        long[] shape = mtcnnUtils.shape(img);
        INDArray ret = mtcnnUtils.imresample(img, (int) Math.ceil(shape[2] * scale), (int) Math.ceil(shape[3] * scale));
        return ret.subi(127.5).muli(0.0078125);
    }

    private INDArray mergeBoxes(INDArray totalBoxes, INDArray boxes) {
        if(boxes.rank() == 1) {
            boxes = Nd4j.expandDims(boxes, 0);
        }
        if (totalBoxes == null) {
            totalBoxes = boxes;
        } else {
            return Nd4j.concat(0, totalBoxes, boxes);
        }
        return totalBoxes;
    }
}
