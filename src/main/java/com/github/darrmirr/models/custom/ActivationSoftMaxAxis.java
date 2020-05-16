package com.github.darrmirr.models.custom;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

public class ActivationSoftMaxAxis extends BaseActivationFunction {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int axis;

	public ActivationSoftMaxAxis(int axis) {
		this.axis = axis;
	}

    public INDArray getActivation(INDArray in, boolean training) {
    	INDArray maxAxis = max(in, null, null, 0);
    	INDArray targetExp = exp(in.subi(maxAxis));
    	INDArray normalize = sum(targetExp, null, null, 0);
        return targetExp.divi(normalize);
    }
    
    private INDArray max(INDArray src, INDArray res, long[] prefix, int index){
    	if(index == axis){
    		return max(src, res, prefix, index + 1);
    	}
    	if(res == null){
    		res = src.dup();
    	}
    	if(prefix == null){
    		prefix = new long[res.rank()];
    	}
    	long[] shape = res.shape();
    	if(index >= shape.length){
    		double max = Long.MIN_VALUE;
    		for(long i = 0;i < shape[axis];i++){
    			prefix[axis] = i;
    			double tmp = res.getDouble(prefix);
    			if(tmp > max){
    				max = tmp;
    			}
    		}
    		for(long i = 0;i < shape[axis];i++){
    			prefix[axis] = i;
    			res.putScalar(prefix, max);
    		}
    	} else {
    		for(int i = 0;i < shape[index];i++){
    			prefix[index] = i;
    			max(src, res, prefix, index+1);
    		}
    	}
    	return res;
    }
    
    private INDArray sum(INDArray src, INDArray res, long[] prefix, int index){
    	if(index == axis){
    		return sum(src, res, prefix, index+1);
    	}
    	if(res == null){
    		res = src.dup();
    	}
    	if(prefix == null){
    		prefix = new long[res.rank()];
    	}
    	long[] shape = res.shape();
    	if(index >= shape.length){
    		double sum = 0;
    		for(long i = 0;i < shape[axis];i++){
    			prefix[axis] = i;
    			sum += res.getDouble(prefix);
    		}
    		for(long i = 0;i < shape[axis];i++){
    			prefix[axis] = i;
    			res.putScalar(prefix, sum);
    		}
    	} else {
    		for(int i = 0;i < shape[index];i++){
    			prefix[index] = i;
    			sum(src, res, prefix, index+1);
    		}
    	}
    	return res;
    }
    
    private INDArray exp(INDArray src){
    	return exp(src, null, 0);
    }
    
    private INDArray exp(INDArray src, long[] prefix, int index){
    	if(prefix == null){
    		prefix = new long[src.rank()];
    	}
    	long[] shape = src.shape();
    	if(index >= shape.length){
    		src.putScalar(prefix, Math.exp(src.getDouble(prefix)));
    	} else {
    		for(int i = 0;i < shape[index];i++){
    			prefix[index] = i;
    			exp(src, prefix, index+1);
    		}
    	}
    	return src;
    }
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray out = getActivation(in, true);
        INDArray x = out.mul(epsilon).sum(1);
        INDArray dLdz = out.mul(epsilon.subColumnVector(x));
        return new Pair<INDArray, INDArray>(dLdz, null);
    }

    @Override
    public String toString() {
        return "SoftMaxDim";
    }

}
