package com.github.darrmirr.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@Component
public class Nd4jUtils {
    private static final Logger logger = LoggerFactory.getLogger(Nd4jUtils.class);

    /**
     * Replace all values in param srcArray where
     * param number2put is greater than corresponding array value
     *
     * if (number2put > arr[x, y]) then arr[x, y] = number2put
     *
     * @param number2put number to replace
     * @param srcArray source array where replacement is performed
     * @return array with replaces values
     */
    public static INDArray maximum(final Number number2put, final INDArray srcArray) {
        var array2Put = Nd4j.zerosLike(srcArray).addi(number2put);
        return maximum(array2Put, srcArray);
    }

    /**
     * Replace all values in param srcArray where
     * param array2Put is greater than corresponding array value
     *
     * if (array2Put[x,y] > arr[x, y]) then arr[x, y] = array2Put[x,y]
     *
     * @param array2Put array to replace
     * @param srcArray source array where replacement is performed
     * @return array with replaces values
     */
    public static INDArray maximum(final INDArray array2Put, final INDArray srcArray) {
        var arrayMask = srcArray.gt(array2Put);
        return srcArray.putWhereWithMask(arrayMask, array2Put);
    }

    /**
     * Replace all values in param srcArray where
     * param number2put is less than corresponding array value
     *
     * if (number2put < arr[x, y]) then arr[x, y] = number2put
     *
     * @param number2put number to replace
     * @param srcArray source array where replacement is performed
     * @return array with replaces values
     */
    public static INDArray minimum(final Number number2put, final INDArray srcArray) {
        var array2Put = Nd4j.zerosLike(srcArray).addi(number2put);
        return minimum(array2Put, srcArray);
    }

    /**
     * Replace all values in param array2Put where
     * param srcArray is less than corresponding array value
     *
     * if (array2Put[x,y] < arr[x, y]) then arr[x, y] = array2Put[x,y]
     *
     * @param array2Put array to replace
     * @param srcArray source array where replacement is performed
     * @return array with replaces values
     */
    public static INDArray minimum(final INDArray array2Put, final INDArray srcArray) {
        var arrayMask = srcArray.lt(array2Put);
        return srcArray.putWhereWithMask(arrayMask, array2Put);
    }

    /**
     * Find element indices at input array according to condition
     *
     * @param array input array of rank 2
     * @param condition condition that would be applied to elements at input array
     * @return array that contains element indices according to condition
     */
    public static INDArray findFitIndexes(INDArray array, final Condition condition) {
        assert array.rank() == 2;

        var indexesList = new ArrayList<INDArray>();
        Shape.iterate(array, coord -> {
            if (condition.apply(array.getDouble(coord[0]))) {
                var indexArray = Nd4j
                        .create(new double[][] { {coord[0][0], coord[0][1] }})
                        .transposei();
                indexesList.add(indexArray);
            }
        });
        if (indexesList.isEmpty()) {
            return null;
        }
        var indexesArray = Nd4j.hstack(indexesList);
        return array.rows() == 1 ? indexesArray.getRow(1) : indexesArray;
    }


    /**
     * Save INDArray to binary file
     *
     * @param indArray array to save
     * @param fileName file path included file name
     */
    public static void toFile(INDArray indArray, String fileName) {
        try(OutputStream outputStream = Files.newOutputStream(Path.of(fileName))){
            Nd4j.write(outputStream, indArray);
        } catch (IOException e) {
            logger.error("error to save indArray", e);
        }
    }

    // resize image in case of it's size is bigger than scaleSize
    public static INDArray scale(INDArray imageMatrix, int scaleSize) {
        long[] shape = imageMatrix.shape();
        if (shape[2] > scaleSize || shape[3] > scaleSize) {
            var scale = shape[2] > shape[3] ? (double) scaleSize / shape[2] : (double) scaleSize / shape[3];
            return Nd4jUtils.imresample(imageMatrix, (int) Math.ceil(shape[2] * scale), (int) Math.ceil(shape[3] * scale));
        }
        return imageMatrix;
    }

    public static INDArray imresample(INDArray img, int hs, int ws) {
        long[] shape = img.shape();
        long h = shape[2];
        long w = shape[3];
        float dx = (float) w / ws;
        float dy = (float) h / hs;
        INDArray im_data = Nd4j.create(new long[] { 1, 3, hs, ws });
        for (int a1 = 0; a1 < 3; a1++) {
            for (int a2 = 0; a2 < hs; a2++) {
                for (int a3 = 0; a3 < ws; a3++) {
                    im_data.putScalar(new long[] { 0, a1, a2, a3 },
                            img.getDouble(0, a1, (long) Math.floor(a2 * dy), (long) Math.floor(a3 * dx)));
                }
            }
        }
        return im_data;
    }

    public List<INDArray> crop(List<BoundBox> boxes, INDArray image) {
        if (boxes == null) {
            return Collections.emptyList();
        }
        return boxes
                .stream()
                .map(box ->
                        crop(box, image))
                .collect(Collectors.toList());
    }

    public INDArray crop(BoundBox box, INDArray image) {
        var x1 = Math.max(box.x1, 0);
        var x2 = Math.max(box.x2, 0);
        var y1 = Math.max(box.y1, 0);
        var y2 = Math.max(box.y2, 0);
        return image.get(all(), all(), interval(y1, y2), interval(x1, x2)).dup();
    }
}
