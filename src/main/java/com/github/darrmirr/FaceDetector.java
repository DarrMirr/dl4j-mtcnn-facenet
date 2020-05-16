package com.github.darrmirr;

import com.github.darrmirr.models.InceptionResNetV1;
import com.github.darrmirr.models.mtcnn.Mtcnn;
import com.github.darrmirr.utils.ImageUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 * Getting Face Features
 *
 * Process consist two stages:
 * 1. Detect faces in image using MTCNN
 * 2. Extract face features from detected face
 *
 */

@Component
public class FaceDetector {
    private static final Logger logger = LoggerFactory.getLogger(FaceDetector.class);
    private NativeImageLoader loader = new NativeImageLoader();
    private Mtcnn mtcnn;
    private ImageUtils imageUtils;
    private InceptionResNetV1 inceptionResNetV1;
    private ComputationGraph faceFeatureExtracter;

    @Autowired
    public FaceDetector(Mtcnn mtcnn, InceptionResNetV1 inceptionResNetV1, ImageUtils imageUtils) {
        this.mtcnn = mtcnn;
        this.imageUtils = imageUtils;
        this.inceptionResNetV1 = inceptionResNetV1;
    }

    @PostConstruct
    public void init() throws IOException {
        faceFeatureExtracter = inceptionResNetV1.getGraph();
    }

    /**
     * Detect faces in image
     *
     * @param image image file
     * @return array of detected images
     * @throws IOException exception while file is read
     */

    public INDArray[] detectFaces(File image) throws IOException {
        var imageMatrix = loader.asMatrix(image);
        return mtcnn.getFaceImages(imageMatrix, 160, 160);
    }

    /**
     * Extract features for each face in array
     *
     * @param faces INDArray represent faces on image (image size depend on model)
     * @return list of face feature vectors
     */
    public List<INDArray> extractFeatures(INDArray[] faces) {
        return Arrays
                .stream(faces)
                .map(faceFeatureExtracter::output)
                .map(output -> output[1])
                .collect(toList());
    }

    /**
     * Method combine detect faces and extract features for each face in array
     *
     * @param image image file
     * @return list of face feature vectors
     * @throws IOException exception while file is read
     */

    public List<INDArray> getFaceFeatures(File image) throws IOException {
        INDArray[] detectedFaces = detectFaces(image);

        if(detectedFaces == null) {
            logger.warn("no face detected in image file:{}", image);
            return null;
        }
        if(detectedFaces.length > 1) {
            logger.warn("{} faces detected in image file:{}, the first detected face will be used.",
                    detectedFaces.length, image);
        }

        if(logger.isDebugEnabled()) {
            logger.debug("save detected face image");
            for (int i = 0; i < detectedFaces.length; i++) {
                imageUtils.toFile(detectedFaces[i], "jpg", image.getName() + "_" + i);
            }
        }
        return extractFeatures(detectedFaces);
    }
}
