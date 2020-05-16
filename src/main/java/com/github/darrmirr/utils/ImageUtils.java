package com.github.darrmirr.utils;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;

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
}
