package com.github.darrmirr.models.mtcnn.networks;

import com.github.darrmirr.models.mtcnn.MtcnnUtils;
import com.github.darrmirr.models.mtcnn.networks.dl4j.RefineNetModel;
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

import static org.nd4j.linalg.factory.Nd4j.shape;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Component
public class RefineNet {
    private static final Logger logger = LoggerFactory.getLogger(RefineNet.class);
    private RefineNetModel refineNetModel;
    private MtcnnUtils mtcnnUtils;
    private ComputationGraph graph;

    @Autowired
    public RefineNet(RefineNetModel refineNetModel, MtcnnUtils mtcnnUtils) {
        this.refineNetModel = refineNetModel;
        this.mtcnnUtils = mtcnnUtils;
    }

    @PostConstruct
    public void init() throws IOException {
        graph = refineNetModel.getGraph();
    }

    /**
     * Stage 2:
     * 01. Pad out-of-bound boxes
     * 02. Feed scaled images into R-Net
     * 03. Gather R-Net output
     * 04. Find indices of bounding boxes with high confidence
     * 05. Non-Maximum Suppression for all boxes
     * 06. Convert bounding box coordinates to “un-scaled image” coordinates
     * 07. Reshape bounding boxes to square
     *
     * @param img input image to find proposal faces
     * @param totalBoxes bouncing boxes retrieve from stage 1 (propose-net)
     * @param threshold threshold for bounding box with low confidence
     * @return array of bounding box face candidates
     */
    public INDArray execute(INDArray img, INDArray totalBoxes, double threshold) {
        logger.debug("refine net : started.");
        INDArray rnetInput = mtcnnUtils.transposeBorder(mtcnnUtils.reshapeAndNorm(img, totalBoxes, 24));
        // 02. Feed scaled images into R-Net
        INDArray rnetOut[] = graph.output(rnetInput);
        // 03. Gather R-Net output
        INDArray score = rnetOut[0].get(all(), point(1)).transposei();
        INDArray reg = rnetOut[1];
        // 04. Find indices of bounding boxes with high confidence
        INDArray ipass = Nd4jUtils.findFitIndexes(score, Conditions.greaterThan(threshold));

        if (ipass == null) {
            return null;
        }
        totalBoxes = mtcnnUtils.squeeze(mtcnnUtils.mergeRegAndScore(totalBoxes, reg, score).get(ipass).dup());
        // or expand dimension to rank 2?
        if (totalBoxes.rank() == 1) {
            return null;
        }
        // 05. Non-Maximum Suppression for all boxes
        totalBoxes = mtcnnUtils.nms(totalBoxes, 0.7, false);
        if (totalBoxes.rank() == 1) {
            return null;
        }
        // 06. Convert bounding box coordinates to “un-scaled image” coordinates
        totalBoxes = mtcnnUtils.bbreg(totalBoxes);
        long[] imgShape = shape(img);
        // 07. Reshape bounding boxes to square
        totalBoxes = mtcnnUtils.rerec(totalBoxes, imgShape[3], imgShape[2]);
        logger.debug("refine net : finished.");
        return totalBoxes;
    }
}
