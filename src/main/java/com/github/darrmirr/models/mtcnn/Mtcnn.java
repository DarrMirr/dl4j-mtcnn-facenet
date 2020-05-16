package com.github.darrmirr.models.mtcnn;

import com.github.darrmirr.utils.ResultBox;
import com.github.darrmirr.models.mtcnn.networks.OutputNet;
import com.github.darrmirr.models.mtcnn.networks.ProposeNet;
import com.github.darrmirr.models.mtcnn.networks.RefineNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Optional;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@Component
public class Mtcnn {
    private static final Logger logger = LoggerFactory.getLogger(Mtcnn.class);
    private ProposeNet proposeNet;
    private RefineNet refineNet;
    private OutputNet outputNet;
    private MtcnnUtils mtcnnUtils;
    private double thresholds[] = { 0.6, 0.7, 0.7 };

    @Autowired
    public Mtcnn(ProposeNet proposeNet, RefineNet refineNet, OutputNet outputNet, MtcnnUtils mtcnnUtils) {
        this.proposeNet = proposeNet;
        this.refineNet = refineNet;
        this.outputNet = outputNet;
        this.mtcnnUtils = mtcnnUtils;
    }

    /**
     * Here’s a short summary of the whole process:
     * Stage 1 - propose net
     * Stage 2 - refine net
     * Stage 3 - output net
     * @param img input image to detect faces in it
     * @return bounded boxes
     */
    public ResultBox[] detectFaces(INDArray img) {
        logger.debug("detectFaces : started");
        var resultBoxes = Optional
                .ofNullable(img)
                .map(image ->
                        proposeNet.execute(image, thresholds[0]))
                .map(proposeBoxes ->
                        refineNet.execute(img, proposeBoxes, thresholds[1]))
                .map(refinedProposeBoxes ->
                        outputNet.execute(img, refinedProposeBoxes, thresholds[2]))
                .map(ResultBox::create)
                .orElse(null);
        logger.debug("detectFaces : finished");
        return resultBoxes;
    }

    public INDArray[] getFaceImages(INDArray img, int width, int height) {
        ResultBox[] boxes = detectFaces(img);
        if(boxes == null) {
            return null;
        }
        INDArray[] ret = new INDArray[boxes.length];
        for(int i = 0;i < ret.length;i++) {
            var x1 = boxes[0].x1 >= 0 ? boxes[0].x1 : 0;
            var x2 = boxes[0].x2 >= 0 ? boxes[0].x2 : 0;
            var y1 = boxes[0].y1 >= 0 ? boxes[0].y1 : 0;
            var y2 = boxes[0].y2 >= 0 ? boxes[0].y2 : 0;
            ret[i] = mtcnnUtils.imresample(img.get(all(), all(), interval(y1, y2), interval(x1, x2)).dup(), height, width);
        }
        return ret;
    }
}
