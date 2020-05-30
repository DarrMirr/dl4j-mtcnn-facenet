package com.github.darrmirr.utils;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class ImageFace {
    private static final Logger log = LoggerFactory.getLogger(ImageFace.class);
    private INDArray imageFace;
    private BoundBox boundBox;
    private INDArray featureVector;

    public ImageFace(INDArray imageFace, BoundBox boundBox) {
        this.imageFace = imageFace;
        this.boundBox = boundBox;
    }

    public INDArray get() {
        return imageFace;
    }

}
