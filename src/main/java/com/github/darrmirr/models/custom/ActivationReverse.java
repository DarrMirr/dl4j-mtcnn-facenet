package com.github.darrmirr.models.custom;

import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class ActivationReverse extends ActivationIdentity {
	
	private static final long serialVersionUID = -3713491302534935272L;


	@Override
    public INDArray getActivation(INDArray in, boolean training) {
		long[] shape = in.shape();
		double d[][] = new double[(int) shape[0]][(int) shape[1]];
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[i].length; j++) {
				d[i][d[i].length - 1 - j] = shape[0] == 1 ? in.getDouble(j) : in.getDouble(i, j);
			}
		}
		return in.assign(Nd4j.create(d));
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        return new Pair<INDArray, INDArray>(getActivation(epsilon, true), null);
    }

    @Override
    public String toString() {
        return "linear";
    }
}
