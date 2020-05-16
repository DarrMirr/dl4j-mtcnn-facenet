package com.github.darrmirr.utils;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@AllArgsConstructor
public class ResultBox {
    public int x1, y1;
    public int x2, y2;

    public static ResultBox[] create(INDArray totalBoxes) {
        if (totalBoxes.rank() == 1) {
            totalBoxes = Nd4j.expandDims(totalBoxes, 0);
        }
        ResultBox[] boxes = new ResultBox[(int) totalBoxes.shape()[0]];
        for (int i = 0; i < boxes.length; i++) {
            boxes[i] = new ResultBox(totalBoxes.getInt(i, 0), totalBoxes.getInt(i, 1), totalBoxes.getInt(i, 2),
                    totalBoxes.getInt(i, 3));
        }
        return boxes;
    }
}
