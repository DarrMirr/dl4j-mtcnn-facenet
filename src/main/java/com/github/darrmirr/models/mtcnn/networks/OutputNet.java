package com.github.darrmirr.models.mtcnn.networks;

import com.github.darrmirr.models.mtcnn.MtcnnUtils;
import com.github.darrmirr.models.mtcnn.networks.dl4j.OutputNetModel;
import com.github.darrmirr.utils.Nd4jUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Component
public class OutputNet {
    private static final Logger logger = LoggerFactory.getLogger(OutputNet.class);
    private OutputNetModel outputNetModel;
    private ComputationGraph graph;
    private MtcnnUtils mtcnnUtils;

    @Autowired
    public OutputNet(OutputNetModel outputNetModel, MtcnnUtils mtcnnUtils) {
        this.outputNetModel = outputNetModel;
        this.mtcnnUtils = mtcnnUtils;
    }

    @PostConstruct
    public void init() throws IOException {
        graph = outputNetModel.getGraph();
    }

    /**
     * Stage 3:
     * 01. Pad out-of-bound boxes
     * 02. Feed scaled images into O-Net
     * 03. Gather O-Net output
     * 04. Find indices of bounding boxes with high confidence
     * 05. Convert bounding box and facial landmark coordinates to “un-scaled image” coordinates
     * 06. Non-Maximum Suppression for all boxes
     *
     * @param img input image to find proposal faces
     * @param totalBoxes bouncing boxes retrieve from stage 2 (refine-net)
     * @param threshold threshold for bounding box with low confidence
     * @return array of bounding box face in image
     */
    public INDArray execute(INDArray img, INDArray totalBoxes, double threshold) {
        logger.debug("output net : started");
        INDArray onetInput = mtcnnUtils.transposeBorder(mtcnnUtils.reshapeAndNorm(img, totalBoxes, 48));
        // 02. Feed scaled images into O-Net
        INDArray onetOut[] = graph.output(onetInput);
        // 03. Gather O-Net output
        var score = onetOut[0].get(all(), point(1)).transposei();
        var reg = onetOut[1];
        // 04. Find indices of bounding boxes with high confidence
        var ipass = Nd4jUtils.findFitIndexes(score, Conditions.greaterThan(threshold));
        if (ipass == null) {
            return null;
        }
        totalBoxes = mtcnnUtils.squeeze(mtcnnUtils.mergeRegAndScore(totalBoxes, reg, score).get(ipass).dup());
        if (totalBoxes.rank() == 1) {
            return null;
        }
        // 05. Convert bounding box and facial landmark coordinates to “un-scaled image” coordinates
        totalBoxes = mtcnnUtils.bbreg(totalBoxes);
        // 06. Non-Maximum Suppression for all boxes
        totalBoxes = mtcnnUtils.nms(totalBoxes, 0.7, true);
        logger.debug("output net : finished");
        return totalBoxes;
    }
}
