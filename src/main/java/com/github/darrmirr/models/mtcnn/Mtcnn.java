package com.github.darrmirr.models.mtcnn;

import com.github.darrmirr.models.mtcnn.networks.OutputNet;
import com.github.darrmirr.models.mtcnn.networks.ProposeNet;
import com.github.darrmirr.models.mtcnn.networks.RefineNet;
import com.github.darrmirr.utils.BoundBox;
import com.github.darrmirr.utils.Nd4jUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Component
public class Mtcnn {
    private static final Logger logger = LoggerFactory.getLogger(Mtcnn.class);
    private ProposeNet proposeNet;
    private RefineNet refineNet;
    private OutputNet outputNet;
    private MtcnnUtils mtcnnUtils;
    private double thresholds[] = { 0.6, 0.7, 0.7 };
    private int optimizedScaleSize = 600;

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
    public List<BoundBox> detectFaces(INDArray img) {
        logger.debug("detectFaces : started");
        var scaledImage = Nd4jUtils.scale(img, optimizedScaleSize);
        var boundBoxes = Optional
                .ofNullable(scaledImage)
                .map(image ->
                        proposeNet.execute(image, thresholds[0]))
                .map(proposeBoxes ->
                        refineNet.execute(scaledImage, proposeBoxes, thresholds[1]))
                .map(refinedProposeBoxes ->
                        outputNet.execute(scaledImage, refinedProposeBoxes, thresholds[2]))
                .map(BoundBox::create)
                .orElse(Collections.emptyList());

        //        var newImageMatrix = imageUtils.drawBoundBox(boundBox, imageMatrix);
        //        imageUtils.toFile(newImageMatrix, "jpg", image.getName());

        int originalHeight = (int) img.shape()[2];
        int originalWidth = (int) img.shape()[3];
        if(originalHeight > optimizedScaleSize || originalWidth > optimizedScaleSize) {
            var scale = originalHeight > originalWidth ? (double) originalHeight / optimizedScaleSize : (double) originalWidth / optimizedScaleSize;
            boundBoxes = reScale(boundBoxes, scale, originalHeight, originalWidth);
        }
        logger.debug("detectFaces : finished");
        return boundBoxes;
    }

    private List<BoundBox> reScale(List<BoundBox> boundBoxes, double scale, int originalHeight, int originalWidth){
        return boundBoxes
                .stream()
                .map(box -> {
                    var originalBbox = mtcnnUtils.scale(box, scale);
                    originalBbox.sourceHeight = originalHeight;
                    originalBbox.sourceWidth = originalWidth;
                    return originalBbox;
                })
                .collect(Collectors.toList());
    }
}