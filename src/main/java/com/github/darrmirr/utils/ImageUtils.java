package com.github.darrmirr.utils;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

/**
 * Utility class to perform operation with image
 */

@Component
public class ImageUtils {
    private static final Logger logger = LoggerFactory.getLogger(ImageUtils.class);
    private Java2DFrameConverter java2DFrameConverter = new Java2DFrameConverter();
    private Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();

    /**
     * Save image to file
     *
     * @param image INDArray image representation
     * @param imageFormat required image format to save
     * @param filePath image file path
     * @return operation result
     * @throws IOException exception while file is saved
     */
    public boolean toFile(INDArray image, String imageFormat, String filePath) throws IOException {
        return toFile(toBufferedImage(image), imageFormat, filePath);
    }

    /**
     * Save image to file
     *
     * @param image buffered image
     * @param imageFormat required image format to save
     * @param filePath image file path
     * @return operation result
     * @throws IOException exception while file is saved
     */
    public boolean toFile(BufferedImage image, String imageFormat, String filePath) throws IOException {
        var outputFile = Paths.get(filePath).toFile();
        logger.info("save image to {}", outputFile.getAbsolutePath());
        return ImageIO.write(image, imageFormat, outputFile);
    }

    /**
     * convert INDArray image representation to Buffered image
     *
     * @param image INDArray image representation
     * @return buffered image
     */
    public BufferedImage toBufferedImage(INDArray image) {
        return java2DFrameConverter.convert(imageLoader.asFrame(image, 8));
    }

    /**
     * Draw bound box on image
     *
     * @param bboxes bound box list to draw
     * @param image source image
     * @return image INDArray with drawn bound boxes
     * @throws IOException during method execution
     */
    public INDArray drawBoundBox(List<BoundBox> bboxes, INDArray image) throws IOException {
        var originalImage = toBufferedImage(image);
        var g2D = originalImage.createGraphics();
        g2D.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2D.setStroke(new BasicStroke(2));
        g2D.setColor(Color.YELLOW);

        bboxes.forEach(bbox ->
                g2D.drawRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1));
        g2D.dispose();
        return imageLoader.asMatrix(java2DFrameConverter.convert(originalImage));
    }

    public INDArray drawBoundBox(BoundBox bbox, INDArray image) throws IOException {
        return drawBoundBox(Collections.singletonList(bbox), image);
    }

    public Frame asFrame(BufferedImage image) {
        return java2DFrameConverter.convert(image);
    }
}
