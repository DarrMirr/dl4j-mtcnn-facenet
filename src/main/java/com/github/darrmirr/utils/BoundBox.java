package com.github.darrmirr.utils;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

@AllArgsConstructor
@NoArgsConstructor
public class BoundBox {
    public int x1, y1;
    public int x2, y2;
    public int sourceWidth;
    public int sourceHeight;

    public BoundBox(int x1, int y1, int x2, int y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    public static List<BoundBox> create(INDArray totalBoxes) {
        if (totalBoxes.rank() == 1) {
            totalBoxes = Nd4j.expandDims(totalBoxes, 0);
        }
        var boxesAmount = (int) totalBoxes.shape()[0];
        List<BoundBox> boxes = new ArrayList<>(boxesAmount);
        for (int i = 0; i < boxesAmount; i++) {
            var box = new BoundBox(
                    totalBoxes.getInt(i, 0),
                    totalBoxes.getInt(i, 1),
                    totalBoxes.getInt(i, 2),
                    totalBoxes.getInt(i, 3)
            );
            boxes.add(box);
        }
        return boxes;
    }
}
