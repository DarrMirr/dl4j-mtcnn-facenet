package com.github.darrmirr.models.mtcnn;

import com.github.darrmirr.utils.ArgSortParam;
import com.github.darrmirr.utils.BoundBox;
import com.github.darrmirr.utils.Nd4jUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;

import static com.github.darrmirr.utils.Nd4jUtils.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

@Component
public class MtcnnUtils {

    /**
     * Non-maximum Suppression (NMS)
     *
     * Description:
     * (source: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
     *
     * Typical Object detection pipeline has one component for generating proposals for classification.
     * Proposals are nothing but the candidate regions for the object of interest.
     * Most of the approaches employ a sliding window over the feature map and assigns foreground/background scores
     * depending on the features computed in that window. The neighbourhood windows have similar scores
     * to some extent and are considered as candidate regions. This leads to hundreds of proposals.
     * As the proposal generation method should have high recall, we keep loose constraints in this stage.
     * However processing these many proposals all through the classification network is cumbersome.
     * This leads to a technique which filters the proposals based on some criteria ( which we will see soon)
     * called Non-maximum Suppression.
     *
     * @param boxes ND array of Proposal boxes
     * @param threshold  overlap threshold
     * @param isMethodMin type
     * @return ND array of filtered proposals D.
     */
    public INDArray nms(INDArray boxes, double threshold, boolean isMethodMin) {
        INDArray boxTmp = boxes.dup().transposei();

        // grab the coordinates of the bounding boxes
        INDArray x1 = boxTmp.get(point(0), all()).dup().reshape(boxes.shape()[0]);
        INDArray y1 = boxTmp.get(point(1), all()).dup().reshape(boxes.shape()[0]);
        INDArray x2 = boxTmp.get(point(2), all()).dup().reshape(boxes.shape()[0]);
        INDArray y2 = boxTmp.get(point(3), all()).dup().reshape(boxes.shape()[0]);

        // compute the area of the bounding boxes and sort the bounding boxes
        INDArray area = x2.sub(x1).addi(1).muli(y2.sub(y1).addi(1));
        INDArray sortedScoreIndex = getScoreSortedIndex(boxes);

        // initialize the list of picked indexes
        var tmpSelectedIndex = new ArrayList<Integer>();
        while (true) {
            // grab the last index in the indexes list and add the
            // index value to the list of picked indexes
            int lastIndex = (int) sortedScoreIndex.length() - 1;
            int highestIndex = sortedScoreIndex.getInt(lastIndex);
            tmpSelectedIndex.add(highestIndex);
            if (lastIndex == 0) {
                break;
            }
            INDArray otherIndexes = sortedScoreIndex.get(interval(0, lastIndex));

            // IoU (Intersection over Union) calculate

            // find the largest (x, y) coordinates for the start of
            // the bounding box and the smallest (x, y) coordinates
            // for the end of the bounding box
            INDArray xx1 = maximum(x1.getDouble(highestIndex), x1.get(otherIndexes));
            INDArray yy1 = maximum(y1.getDouble(highestIndex), y1.get(otherIndexes));
            INDArray xx2 = minimum(x2.getDouble(highestIndex), x2.get(otherIndexes));
            INDArray yy2 = minimum(y2.getDouble(highestIndex), y2.get(otherIndexes));

            // compute the width and height of the bounding box
            INDArray w = maximum(0.0, xx2.sub(xx1).addi(1));
            INDArray h = maximum(0.0, yy2.sub(yy1).addi(1));
            INDArray interArea = w.mul(h);

            // compute the ratio of overlap
            INDArray overlap;
            if (isMethodMin) {
                overlap = interArea.divi(minimum(area.getDouble(highestIndex), area.get(otherIndexes)));
            } else {
                overlap = interArea.divi(area.get(otherIndexes).addi(area.getDouble(highestIndex)).subi(interArea));
            }
            INDArray tmpFitIdx = findFitIndexes(overlap, Conditions.lessThanOrEqual(threshold));
            if (tmpFitIdx == null) {
                break;
            }
            sortedScoreIndex = sortedScoreIndex.get(tmpFitIdx);
        }
        INDArray retIdx = Nd4j.create(new int[] { tmpSelectedIndex.size() });
        for (int i = 0; i < tmpSelectedIndex.size(); i++) {
            retIdx.putScalar(i, tmpSelectedIndex.get(i));
        }

        INDArray ret = boxes.get(retIdx);
        ret = squeeze(ret);
        if(ret.rank() == 1) {
            assert ret.shape()[0] == 9;
        } else {
            assert ret.shape()[1] == 9;
        }
        return ret;
    }

    /**
     * Sort scores of the bounding boxes
     *
     * @param boxes ND array of bounding boxes, where column 4 contains scores
     * @return array contains scores order by desc (? or asc)
     */

    protected INDArray getScoreSortedIndex(INDArray boxes) {
        int scoreIndex = 4;
        long boxShape[] = boxes.shape();
        ArgSortParam arr[] = new ArgSortParam[(int) boxShape[0]];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new ArgSortParam(boxes.getDouble(i, scoreIndex), i);
        }
        Arrays.sort(arr);
        INDArray ret = Nd4j.create(arr.length);
        for (int i = 0; i < arr.length; i++) {
            ret.putScalar(i, arr[i].getIndex());
        }
        return ret;
    }

    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param indArray array to get rid of any singleton dimensions
     * @return return array with removed singleton dimensions
     */
    public INDArray squeeze(INDArray indArray) {
        var squeezedShape = Shape.squeeze(indArray.shape());
        return indArray.reshape(squeezedShape);
    }

    // Reorder image dimensions from [] to [dimension,channel,w,h]
    public INDArray transposeBorder(INDArray arr) {
        return arr.permutei(0,1,3,2);
    }

    /**
     * Convert the bounding box coordinates to coordinates of the actual image.
     * Right now, the coordinates of each bounding box is a value between 0 and 1,
     * with (0,0) as the top left corner of the 12 x 12 kernel and (1,1) as the bottom right corner.
     * By multiplying the coordinates by the actual image width and height,
     * we can convert the bounding box coordinates to the standard, real-sized image coordinates
     *
     * @param totalBoxes bounding boxes from neuron network output
     * @return regularized bounding boxes coordinates to real-size image one
     */
    public INDArray bbreg(INDArray totalBoxes) {
        INDArray x1 = totalBoxes.get(all(), point(0)).dup();
        INDArray y1 = totalBoxes.get(all(), point(1)).dup();
        INDArray x2 = totalBoxes.get(all(), point(2)).dup();
        INDArray y2 = totalBoxes.get(all(), point(3)).dup();
        INDArray regw = x2.sub(x1);
        INDArray regh = y2.sub(y1);
        totalBoxes.put(new INDArrayIndex[] { all(), point(0) },
                x1.addi(totalBoxes.get(all(), point(5)).dup().muli(regw)));
        totalBoxes.put(new INDArrayIndex[] { all(), point(1) },
                y1.addi(totalBoxes.get(all(), point(6)).dup().muli(regh)));
        totalBoxes.put(new INDArrayIndex[] { all(), point(2) },
                x2.addi(totalBoxes.get(all(), point(7)).dup().muli(regw)));
        totalBoxes.put(new INDArrayIndex[] { all(), point(3) },
                y2.addi(totalBoxes.get(all(), point(8)).dup().muli(regh)));
        return totalBoxes;
    }

    /*Convert the bbox into square*/
    public INDArray rerec(INDArray totalBoxes, double imgW, double imgH) {
        INDArray x1 = totalBoxes.get(all(), point(0)).dup();
        INDArray y1 = totalBoxes.get(all(), point(1)).dup();
        INDArray x2 = totalBoxes.get(all(), point(2)).dup();
        INDArray y2 = totalBoxes.get(all(), point(3)).dup();
        INDArray w = x2.sub(x1);
        INDArray h = y2.sub(y1);
        INDArray regl = maximum(w, h.dup());
        INDArray lossW = regl.sub(w).muli(0.5);
        INDArray lossH = regl.sub(h).muli(0.5);
        totalBoxes.put(new INDArrayIndex[] { all(), point(0) }, Transforms.floor((maximum(0.0, x1.subi(lossW)))));
        totalBoxes.put(new INDArrayIndex[] { all(), point(1) }, Transforms.floor((maximum(0.0, y1.subi(lossH)))));
        totalBoxes.put(new INDArrayIndex[] { all(), point(2) }, Transforms.floor((minimum(imgW, x2.addi(lossW)))));
        totalBoxes.put(new INDArrayIndex[] { all(), point(3) }, Transforms.floor((minimum(imgH, y2.addi(lossH)))));
        return totalBoxes;
    }

    public INDArray mergeRegAndScore(INDArray totalBoxes, INDArray reg, INDArray score) {
        mergeReg(totalBoxes, reg);
        mergeScore(totalBoxes, score);
        return totalBoxes;
    }

    public INDArray mergeScore(INDArray totalBoxes, INDArray score) {
        merge(totalBoxes, score.dup().transposei(), new INDArrayIndex[] { all(), point(4) });
        return totalBoxes;
    }

    private INDArray mergeReg(INDArray totalBoxes, INDArray reg) {
        merge(totalBoxes, reg, new INDArrayIndex[] { all(), interval(5, 9) });
        return totalBoxes;
    }

    private void merge(INDArray sourceArray, INDArray array2Merge, INDArrayIndex[] indices) {
        sourceArray.put(indices, array2Merge);
    }

    public INDArray reshapeAndNorm(INDArray img, INDArray totalBoxes, int border) {
        long[] boxShape = Nd4j.shape(totalBoxes);
        INDArray ret = Nd4j.create(boxShape[0], img.shape()[1], border, border);
        for (int i = 0; i < boxShape[0]; i++) {
            INDArray reshapedImg = Nd4jUtils.imresample(
                    img.get(all(), all(), interval(totalBoxes.getInt(i, 1), totalBoxes.getInt(i, 3)),
                            interval(totalBoxes.getInt(i, 0), totalBoxes.getInt(i, 2))).dup(),
                    border, border);
            ret.put(new INDArrayIndex[] { point(i), all(), all(), all() }, reshapedImg);
        }
        return ret.subi(127.5).muli(0.0078125);
    }

    public long[] shape(INDArray img) {
        long[] ret = img.shape();
        if (ret.length == 2 && ret[0] == 1) {
            ret = new long[] { ret[1] };
        }
        if (ret.length == 0) {
            ret = new long[] { 1 };
        }
        return ret;
    }

    public INDArray generateBox(INDArray score, INDArray reg, double threshold, double scale) {
        assert reg.rank() == 4 && reg.shape()[0] == 1;
        // 05. Find indices of bounding boxes with high confidence (>= threshold)
        INDArray fitIndexes = Nd4jUtils.findFitIndexes(score, Conditions.greaterThanOrEqual(threshold));
        if (fitIndexes == null) {
            return null;
        }
        //Convert 12 x 12 kernel coordinates to “un-scaled image” coordinates
        int stride = 2, cellSize = 12;
        INDArray upperLeft = Transforms.floor(fitIndexes.mul(stride).addi(1).divi(scale).transposei()),
                bottomRight = Transforms.floor(fitIndexes.mul(stride).addi(cellSize).divi(scale).transpose());

        return Nd4j.hstack(upperLeft, bottomRight, score.get(fitIndexes).transposei(),
                reg.get(point(0), point(3), all(), all()).get(fitIndexes).transposei(),
                reg.get(point(0), point(2), all(), all()).get(fitIndexes).transposei(),
                reg.get(point(0), point(1), all(), all()).get(fitIndexes).transposei(),
                reg.get(point(0), point(0), all(), all()).get(fitIndexes).transposei());
    }

    public BoundBox scale(BoundBox box, double scale) {
        BoundBox originalBox = new BoundBox();
        originalBox.x1 = (int) Math.ceil(box.x1 * scale);
        originalBox.y1 = (int) Math.ceil(box.y1 * scale);
        originalBox.x2 = (int) Math.ceil(box.x2 * scale);
        originalBox.y2 = (int) Math.ceil(box.y2 * scale);
        return originalBox;
    }

    public BoundBox scale(BoundBox box, int size2scale) {
        double scale = box.sourceWidth > box.sourceHeight ?
                (double) size2scale / box.sourceWidth : (double) size2scale / box.sourceHeight;
        return scale(box, scale);
    }
}
