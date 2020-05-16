package com.github.darrmirr.models.custom;

import com.github.darrmirr.utils.Nd4jUtils;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.PReLUParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
//import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class PReLUNormLayer extends BaseLayer<PReLUNorm> {
	private static final long serialVersionUID = 1L;

	// for beta-2 version
    public PReLUNormLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr mgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, mgr);

        INDArray in;
        if (training) {
            in = mgr.dup(ArrayType.ACTIVATIONS, input, input.ordering());
        } else {
            in = mgr.leverageTo(ArrayType.ACTIVATIONS, input);
        }

        INDArray alpha = getParam(PReLUParamInitializer.WEIGHT_KEY);
        IActivation activation = new ActivationPReLU(alpha, null);
        if(in.rank() == 4){
        	return activate(in, activation, training);
        } else {
        	return activation.getActivation(in, training);
        }
    }

	protected INDArray activate(INDArray in, IActivation activation, boolean training) {
        long[] inshape = in.shape();
        // rank 2 is need to activation function
		long[] shape4ActFunc = new long[] {inshape[0]*inshape[2]*inshape[3], inshape[1]};
//		INDArray tmp = Shape.newShapeNoCopy(in.permutei(0,2,3,1), shape4ActFunc, false);
        var tmp = in.permutei(0,2,3,1).reshape(shape4ActFunc);
        tmp = activation.getActivation(tmp, training);

		long[] shape4Layer = new long[] {inshape[0], inshape[2], inshape[3], inshape[1]};
		return Shape.newShapeNoCopy(tmp, shape4Layer, false).permutei(0,3,1,2);
	}

	@Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray layerInput = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, input, input.ordering());

        INDArray alpha = getParam(PReLUParamInitializer.WEIGHT_KEY);
        IActivation prelu = new ActivationPReLU(alpha, null);

        Pair<INDArray, INDArray> deltas = prelu.backprop(layerInput, epsilon);
        INDArray delta = deltas.getFirst();
        INDArray weightGradView = deltas.getSecond();

        delta = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta);  //Usually a no-op (except for perhaps identity)
        delta = backpropDropOutIfPresent(delta);
        Gradient ret = new DefaultGradient();
        ret.setGradientFor(PReLUParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        return new Pair<Gradient, INDArray>(ret, delta);
    }


    public boolean isPretrainLayer() {
        return false;
    }

}
