package com.github.darrmirr.models.custom;

import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

public class ActivationLinear extends ActivationIdentity {
	
	private static final long serialVersionUID = -3713491302534935272L;
	private double scale = 1;
	private double bias = 0;
	
	
	public ActivationLinear(double scale, double bias) {
		this.scale = scale;
		this.bias = bias;
	}
	
	public ActivationLinear(double scale) {
		this.scale = scale;
	}
	
	public ActivationLinear() {
	}

	@Override
    public INDArray getActivation(INDArray in, boolean training) {
        return in.muli(scale).addi(bias);
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        return new Pair<INDArray, INDArray>(epsilon.muli(scale), null);
    }

    @Override
    public String toString() {
        return "linear";
    }
}
