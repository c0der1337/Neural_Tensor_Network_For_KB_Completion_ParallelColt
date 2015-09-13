

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.google.common.base.Functions;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.impl.DenseIntMatrix1D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.stanford.nlp.ling.tokensregex.types.AssignableExpression;
import edu.stanford.nlp.optimization.DiffFunction;
import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.IPair;

public class NTN implements IDifferentiableFn, DiffFunction {
	
	int embeddingSize; 			// 100 size of a single word vector
	int numberOfEntities;
	int numberOfRelations;
	int numOfWords; 			// number of word vectors
	int batchSize; 				// training batch size
	int sliceSize;				// 3 number of slices in tensor
	double lamda;				// regulariization parameter
	DoubleMatrix1D theta_initalP;// stacked paramters after initlization of NTN
	//DoubleMatrix1D theta_initalP2;
	int dimension_for_minimizer;// dimensions / size of theta for the minimizer
	String activationFunc;		// "tanh" or "sigmoid"
	String weight_path;
	//Network Parameters - Integer identifies the relation, there are different parameters for each relation
	HashMap<Integer, TensorP> wP;
	HashMap<Integer, DoubleMatrix2D> vP;
	HashMap<Integer, DoubleMatrix2D> bP;
	HashMap<Integer, DoubleMatrix2D> uP;
	DoubleMatrix2D wordvectorsP;
	IPair<double[], Object> theta_CostIP= null;
	IPair<Double, double[]> cost_thetaIP = BasicPair.make( 100.0, new double[4] );
	
	int update;							
	DataFactory tbj;

	NTN(	int embeddingSize, 			// 100 size of a single word vector
			int numberOfEntities,		
			int numberOfRelations,		// number of different relations
			int numberOfWords,			// number of word vectors
			int batchSize, 				// training batch size original: 20.000
			int sliceSize, 				// 3 number of slices in tensor
			String activation_function,	// 0 - tanh, 1 - sigmoid
			DataFactory tbj,			// data management unit
			double lamda,				// regulariization parameter
			String weight_path			// path for loading pre-trained weights
								) throws IOException{  				
		
		// Initialize the parameters of the network
		wP = new HashMap<Integer, TensorP>();
		vP = new HashMap<Integer, DoubleMatrix2D>();
		bP = new HashMap<Integer, DoubleMatrix2D>();
		uP = new HashMap<Integer, DoubleMatrix2D>();
		
		this.embeddingSize = embeddingSize;
		this.numberOfEntities = numberOfEntities;
		this.numberOfRelations = numberOfRelations;
		this.batchSize = batchSize;
		this.sliceSize = sliceSize;
		this.lamda = lamda;
		this.numOfWords = numberOfWords;
		this.activationFunc = activation_function;
		update = 0;
		double r = 1 / Math.sqrt(2*embeddingSize); // r is used for a better initialization of w
		
		for (int i = 0; i < numberOfRelations; i++) {
			TensorP tenP = new TensorP(embeddingSize, embeddingSize, sliceSize, 2);
			for (int j = 0; j < sliceSize; j++) {
				//tenP.setSlice(j, tenP.getSlice(j).assign(DoubleFunctions.mult( (2 * r - r))));
				tenP.setSlice(j, tenP.getSlice(j));
			}
			wP.put(i, tenP);
			vP.put(i, new DenseDoubleMatrix2D(2*embeddingSize,sliceSize).assign(0));
			//vP.put(i, DoubleFactory2D.dense.random(2*embeddingSize,sliceSize));
			bP.put(i, new DenseDoubleMatrix2D(1,sliceSize).assign(0));
			//bP.put(i, DoubleFactory2D.dense.random(1,sliceSize));
			uP.put(i, new DenseDoubleMatrix2D(sliceSize,1).assign(1));
			//uP.put(i, DoubleFactory2D.dense.random(sliceSize,1));
		}
		
		// Initialize WordVectors via loaded Vectors
		wordvectorsP = new DenseDoubleMatrix2D(embeddingSize,numOfWords).assign(tbj.getWordVectorMaxtrixLoaded().transpose().data().asDouble());
		wordvectorsP.assign(DoubleFunctions.mult(0.1));
		// For random wordVector initialization:
		r = 0.001; //Socher:
		//wordvectorsP = DoubleFactory2D.dense.random(embeddingSize, numOfWords).assign(DoubleFunctions.mult(2*r-r));
		//wordvectorsP.assign(DoubleFunctions.mult(2*r-r));
		r = 0.0001; //Python:
		//wordvectorsP = DoubleFactory2D.dense.random(embeddingSize, numOfWords).assign(DoubleFunctions.mult(2*r-r));
		
		// Unroll the parameters into a vector
		theta_initalP = parametersToStackP(wP, vP, bP, uP, wordvectorsP);
		
		//LOAD WEIGHTS
		if (!weight_path.equals("")) {
			//stackToParametersP(new DenseDoubleMatrix1D(Nd4j.readTxt("C://Users//Patrick//Documents//master arbeit//weights//python//4Slice//mul_UKDEIT//theta_075_mulUKDEIT_MIX2.out", ",").data().asDouble()));
			stackToParametersP(new DenseDoubleMatrix1D(Nd4j.readTxt(weight_path, ",").data().asDouble()));
			System.out.println("weights loaded from: "+ weight_path);
		}
		
		//write:
		//Convert solution matrix back into an INDArray
		/*INDArray wordvectors = Nd4j.create(wordvectorsP.rows(),wordvectorsP.columns());
		for (int i = 0; i < wordvectorsP.rows(); i++) { 
			for (int j = 0; j < wordvectorsP.columns(); j++) {
				wordvectors.put(i, j, wordvectorsP.get(i, j));
			}
		}
				
		Nd4j.writeTxt(wordvectors, "C://Users//Patrick//Documents//master arbeit//original_code//data//WordnetUKDE//wordvecmatrixUK_trained_300.txt", ",");
		*/
		theta_initalP = parametersToStackP(wP, vP, bP, uP, wordvectorsP);
		dimension_for_minimizer = (int)theta_initalP.size();
	}
	
	/**
	 * Returns the cost/loss for the current parameters and return optimized paramters 
	 * @param  _theta  	an double array with the flattened parameters of the network
	 * @return      	an IPair that contains of cost and optimized network paramters
	 * @see         ...
	 */
	@Override
	public IPair<Double, double[]> computeAt(double[] _theta) {
		if (java.util.Arrays.equals(cost_thetaIP.getSecond(), _theta)) {
			System.out.println("Cost: " + cost_thetaIP.getFirst());
		}else{
			cost_thetaIP = costfunction(_theta);
		}
		return costfunction(_theta);
	}

	@Override
	public int getDimension() {
		return dimension_for_minimizer;
	}
	
	private DoubleMatrix1D parametersToStackP(HashMap<Integer, TensorP> _w, HashMap<Integer, DoubleMatrix2D> _v, HashMap<Integer, DoubleMatrix2D> _b, HashMap<Integer, DoubleMatrix2D> _u, DoubleMatrix2D _wordvectors){
		DoubleFactory1D F = DoubleFactory1D.dense;
		DoubleMatrix1D theta_return = new DenseDoubleMatrix1D(0);
		
		for (int j = 0; j < wP.size(); j++) {
			// concatenate to stack
			theta_return = F.append(theta_return, _w.get(j).as1DArrayLikeNumpy());
			//System.out.println("size: "+wP.get(j).as1DArrayLikeNumpy().size());
			//theta_return = F.append(theta_return, _w.get(j).as1DArrayLikeNumpy());
			/*for (int i = 0; i < sliceSize; i++) {
				theta_return = F.append(theta_return, _w.get(j).getSlice(i).vectorize());
			}*/
		}

		//v:
		for (int j = 0; j < vP.size(); j++) {
			// concatenate to stack
			theta_return = F.append(theta_return, _v.get(j).viewDice().vectorize());
		}
		//System.out.println("sum: "+theta_return.zSum()+"| size: "+theta_return.size());
		
		//b:
		for (int j = 0; j < bP.size(); j++) {
			// concatenate to stack
			theta_return = F.append(theta_return, _b.get(j).vectorize());
		}
		//System.out.println("sum: "+theta_return.zSum()+"| size: "+theta_return.size());
		
		//U:
		for (int j = 0; j < uP.size(); j++) {
			// concatenate to stack
			theta_return = F.append(theta_return, _u.get(j).vectorize());
		}
		//System.out.println("sum: "+theta_return.zSum()+"| size: "+theta_return.size());
		//Word Vectors:
		theta_return = F.append(theta_return, _wordvectors.viewDice().vectorize());

		//System.out.println("sum: "+theta_return.zSum()+"| size: "+theta_return.size());
		
		return theta_return;		
	}
	
	private void stackToParametersP(DoubleMatrix1D theta){
		//System.out.println("sum: "+theta.zSum()+"| size: "+theta.size());
		int readposition = 0;
		int readposition2 = 0;
		//int w_size = embeddingSize*embeddingSize*sliceSize; // number of values for paramter w for one relation
		int w_size = embeddingSize*embeddingSize;
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		int wordvectors_size = embeddingSize*numOfWords;
		
		DoubleMatrix1D thetaP = theta;
		
		//load w:
		for (int r = 0; r < wP.size(); r++) {
			//System.out.println("thetaP.viewPart(readposition2, w_size*sliceSize): "+thetaP.viewPart(readposition2, w_size*sliceSize).size());
			wP.get(r).fill(thetaP.viewPart(readposition2, w_size*sliceSize));
			readposition2 = readposition2+w_size*sliceSize;
			for (int j = 0; j < sliceSize; j++) {
				//wPNP.get(r).setSlice(j, thetaP.viewPart(readposition, w_size).reshape(embeddingSize, embeddingSize));	
				//wP.get(r).setSlice(j, thetaP.viewPart(readposition, w_size).reshape(embeddingSize, embeddingSize));
				readposition = readposition+w_size;
			}
			//System.out.println("wP: "+wP.get(r).getSlice(0).viewPart(0, 0, 5, 5));
			//System.out.println("ParamToStackMethod: r: "+r+" w.size:"+w.size());
		}
		//load v:
			for (int r = 0; r < vP.size(); r++) {
				//vP.put(r,thetaP.viewPart(readposition, v_size).reshape( sliceSize,2* embeddingSize).viewDice());
				vP.put(r,thetaP.viewPart(readposition, v_size).reshape(sliceSize,2* embeddingSize).viewDice());
				readposition = readposition+v_size;
				//System.out.println(r+"v org: "+vPNP.get(r).viewPart(0, 0, 3, 3));
				//System.out.println(r+"v: "+ vP.get(r).viewPart(0, 0, 3, 3));
			}
			//System.out.println(" here vP.get(1).viewPart(0,0,3, 3): "+vP.get(1).viewPart(0,0,3, 3));
			//System.out.println("vP: "+vP);
			//System.out.println("+++b:");
			//load b:
			for (int r = 0; r < bP.size(); r++) {
				//bP.put(r,thetaP.viewPart(readposition, b_size).reshape(1, sliceSize));
				bP.put(r,thetaP.viewPart(readposition, b_size).reshape(1, sliceSize));
				readposition = readposition+b_size;
				//System.out.println(r+"b org: "+bPNP.get(r).viewPart(0, 0, 1, 3));
				//System.out.println(r+"b: "+ bP.get(r).viewPart(0, 0, 1, 3));
				//System.out.println("bP: "+bP.get(r));
			}
			
			//System.out.println("+++u:");
			//load u:
			for (int r = 0; r < uP.size(); r++) {
				//uP.put(r,thetaP.viewPart(readposition, u_size).reshape(sliceSize,1));
				uP.put(r,thetaP.viewPart(readposition, u_size).reshape(sliceSize,1));
				readposition = readposition+u_size;
				//System.out.println(r+"u org: "+uPNP.get(r).viewPart(0, 0, 3, 1));
				//System.out.println(r+"u: "+ uP.get(r).viewPart(0, 0, 3, 1));
				//System.out.println("uP: "+uP.get(r));
			}
			
			//load word vectors:
			//wordvectorsP.assign(thetaP.viewPart(readposition, wordvectors_size).reshape(numOfWords,embeddingSize).viewDice());
			wordvectorsP.assign(thetaP.viewPart(readposition, wordvectors_size).reshape(numOfWords,embeddingSize).viewDice());
			//System.out.println("wordvecs org: "+wordvectorsPNP.viewPart(0, 0, 10, 10));
			//System.out.println("wordvecs: "+ wordvectorsP.viewPart(0, 0, 10, 10));
			
	}
	public double[] getTheta_inital() {
		return theta_initalP.toArray();
	}
	
	public INDArray activationDifferential(INDArray activation){
		//for a sigmoid activation function:
		//Ableitung der sigmoid function f(z) -> f'(z) -> (z * (1 - z))
		//Ableitung der tanh function f(z) -> f'(z) -> (1-tanh²(z))	
		//System.out.println(activationFunc+ " | "+activation);
			//System.out.println("X"+));
		//System.out.println("Transforms.pow(activation, 2): "+activation.mul(activation));
			//return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, activation).derivative());
		if (activationFunc.equals("tanh")) {
			return Nd4j.ones(activation.rows(), activation.columns()).sub(activation.mul(activation));
		}	else {
			return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, activation).derivative());
		}
			
		
	}

	
	public DataFactory getDatafactory() {
		return tbj;
	}

	public void connectDatafactory(DataFactory tbj) {
		this.tbj = tbj;
	}

	@Override
	public int domainDimension() {
		return dimension_for_minimizer;
	}

	@Override
	public double valueAt(double[] theta) {
		if (theta.equals(theta_CostIP.getFirst())) {
			//System.out.println("Cost: " +theta_CostIP.getSecond());
			return (double) theta_CostIP.getSecond();	
		}else {
			return computeAt(theta).getFirst();
		}
	}

	@Override
	public double[] derivativeAt(double[] theta) {
		IPair result = computeAt(theta);
		theta_CostIP = BasicPair.make( theta, result.getFirst() );
		return (double[]) result.getSecond();
	}
	
	
	public int[] INDArrayToIntArray(INDArray input) {
		int[] x = new int[input.length()];
		
		for (int i = 0; i < input.length(); i++) {
			x[i] = (int)input.getDouble(i);
		}
        return x;
	}
	
	public void thetaEqualsNumpy(DoubleMatrix1D _theta, String path){
		//load correct data from numpy
		HashMap<Integer, TensorP> wPNP_valid= new HashMap<Integer, TensorP>();
		HashMap<Integer, DoubleMatrix2D> vPNP_valid= new HashMap<Integer, DoubleMatrix2D>();
		HashMap<Integer, DoubleMatrix2D> bPNP_valid= new HashMap<Integer, DoubleMatrix2D>();
		HashMap<Integer, DoubleMatrix2D> uPNP_valid= new HashMap<Integer, DoubleMatrix2D>();
		DoubleMatrix2D wordvectorsPNP_valid = new DenseDoubleMatrix2D(embeddingSize,numOfWords);
		
		HashMap<Integer, TensorP> w_test= new HashMap<Integer, TensorP>();
		HashMap<Integer, DoubleMatrix2D> v_test= new HashMap<Integer, DoubleMatrix2D>();
		HashMap<Integer, DoubleMatrix2D> b_test = new HashMap<Integer, DoubleMatrix2D>();
		HashMap<Integer, DoubleMatrix2D> u_test= new HashMap<Integer, DoubleMatrix2D>();
		DoubleMatrix2D wordvectorsPNP_test= new DenseDoubleMatrix2D(embeddingSize,numOfWords);
		
		for (int i = 0; i < numberOfRelations; i++) {
			wPNP_valid.put(i, new TensorP(100, 100, 3, 0));
			vPNP_valid.put(i, new DenseDoubleMatrix2D(2*embeddingSize,sliceSize));
			bPNP_valid.put(i, new DenseDoubleMatrix2D(1,sliceSize));
			uPNP_valid.put(i, new DenseDoubleMatrix2D(sliceSize,1));
			
			w_test.put(i, new TensorP(100, 100, 3, 0));
			v_test.put(i, new DenseDoubleMatrix2D(2*embeddingSize,sliceSize));
			b_test.put(i, new DenseDoubleMatrix2D(1,sliceSize));
			u_test.put(i, new DenseDoubleMatrix2D(sliceSize,1));
		}

		
		//read in
		FileReader fr = null;
		try {
			fr = new FileReader(path);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    BufferedReader br = new BufferedReader(fr);
	    String line = null;
		try {
			line = br.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    double[] input = new double[6805912];
	    int count = 0;
	    while (line != null) {
	    	if (count==0) {
				System.out.println("line:"+line);
			}
	    	//System.out.println(line);
	    	/*for (int j = 0; j <line.split(",").length; j++) {
						input[j] = Double.parseDouble(line.split(",")[j]);
						
						
	    	}*/
	    	input[count] = Double.parseDouble(line);
	    	count++;
	    	try {
				line = br.readLine();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    }
	    System.out.println("Counter: "+count);
		
	    
		int readposition = 0;
		int readposition2 = 0;
		int w_size = embeddingSize*embeddingSize;
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		int wordvectors_size = embeddingSize*numOfWords;
		
		DoubleMatrix1D thetaNP = new DenseDoubleMatrix1D((int)_theta.size()).assign(input);
		/*for (int i = 0; i < thetaNP.size(); i++) {
			if (thetaNP.get(i)== _theta.get(i)) {
				System.out.println("CORRECT!");
			}
		}
		//DoubleMatrix2D thetaNP2D = thetaNP.reshape(1, (int)thetaNP.size());
		System.out.println(equalsColtMatrixes2D(thetaNP.reshape(1, (int)thetaNP.size()), _theta.reshape(1, (int)_theta.size())));*/
		
		//load w:
		for (int r = 0; r < wP.size(); r++) {
			wPNP_valid.get(r).fillLikeNumpy(thetaNP.viewPart(readposition2, w_size*sliceSize));
			readposition2 = readposition2+w_size*sliceSize;
			
			for (int j = 0; j < sliceSize; j++) {
				/*
				wPNP_valid.get(r).setSlice(j, thetaNP.viewPart(readposition, w_size).reshape(embeddingSize, embeddingSize).viewDice());*/
				w_test.get(r).setSlice(j, _theta.viewPart(readposition, w_size).reshape(embeddingSize, embeddingSize).viewDice());
				readposition = readposition+w_size;
				//System.out.println(j+":"+wPNP_valid.get(r).getSlice(j).get(10, 10)+" vs "+w_test.get(r).getSlice(j).get(10, 10));
				//System.out.println(equalsColtMatrixes2D(wPNP_valid.get(r).getSlice(j), w_test.get(r).getSlice(j), true));
				System.out.println(equalsColtMatrixes2D(wPNP_valid.get(r).getSlice(j), w_test.get(r).getSlice(j)));
				w_test.get(r).setSlice(j,correctColtMatrixes2D(wPNP_valid.get(r).getSlice(j), w_test.get(r).getSlice(j)));
			}
			
			//System.out.println("ParamToStackMethod: r: "+r+" w.size:"+w.size());

		}
		
		System.out.println("+++v:");
		//load v:
		for (int r = 0; r < vP.size(); r++) {
			vPNP_valid.put(r,thetaNP.viewPart(readposition, v_size).reshape( sliceSize,2* embeddingSize).viewDice());
			v_test.put(r,_theta.viewPart(readposition, v_size).reshape(2* embeddingSize, sliceSize));
			readposition = readposition+v_size;
			System.out.println(equalsColtMatrixes2D(vPNP_valid.get(r), v_test.get(r)));
			v_test.put(r,correctColtMatrixes2D(vPNP_valid.get(r), v_test.get(r)));
		}
		System.out.println("+++b:");
		//load b:
		for (int r = 0; r < bP.size(); r++) {
			bPNP_valid.put(r,thetaNP.viewPart(readposition, b_size).reshape(1, sliceSize));
			b_test.put(r,_theta.viewPart(readposition, b_size).reshape(1, sliceSize));
			readposition = readposition+b_size;
			System.out.println(equalsColtMatrixes2D(bPNP_valid.get(r), b_test.get(r)));
			b_test.put(r,correctColtMatrixes2D(bPNP_valid.get(r), b_test.get(r)));
		}
		System.out.println("+++u:");
		//load u:
		for (int r = 0; r < uP.size(); r++) {
			uPNP_valid.put(r,thetaNP.viewPart(readposition, u_size).reshape(sliceSize,1));
			u_test.put(r,_theta.viewPart(readposition, u_size).reshape(sliceSize,1));
			readposition = readposition+u_size;
			System.out.println(equalsColtMatrixes2D(uPNP_valid.get(r), u_test.get(r)));
			u_test.put(r,correctColtMatrixes2D(uPNP_valid.get(r), u_test.get(r)));
		}
		System.out.println("+++ wordvecs:");
		//load word vectors:
		wordvectorsPNP_valid.assign(thetaNP.viewPart(readposition, embeddingSize*numOfWords).reshape(numOfWords,embeddingSize).viewDice());
		wordvectorsPNP_test.assign(_theta.viewPart(readposition, wordvectors_size).reshape(embeddingSize,numOfWords));
		System.out.println(equalsColtMatrixes2D(wordvectorsPNP_valid, wordvectorsPNP_test));
		System.out.println(wordvectorsPNP_valid.get(90, 100)+" | "+wordvectorsPNP_test.get(90, 100));
		System.out.println(wordvectorsPNP_valid.get(50, 1000)+" | "+wordvectorsPNP_test.get(50, 100));
		System.out.println(wordvectorsPNP_valid.get(70, 30000)+" | "+wordvectorsPNP_test.get(70, 30000));
		correctColtMatrixes2D(wordvectorsPNP_valid, wordvectorsPNP_test);
		

	}
	public boolean equalsColtMatrixes2D(DoubleMatrix2D validMatrix, DoubleMatrix2D testMatrix){
		boolean equal=true;
		int count_unequal=0;
		double difference = 0;
		String correct="";
		String incorrect="";
		int parameterwithoutWV = 4*(100*100+2*100+3+3);
		for (int i = 0; i < validMatrix.rows(); i++) {
			for (int j = 0; j < validMatrix.columns(); j++) {
				if (validMatrix.get(i, j) != testMatrix.get(i, j)) {
					//System.out.println(i+","+j+": "+validMatrix.get(i, j)+"|vs|"+testMatrix.get(i, j)+"Diff: "+Math.abs(validMatrix.get(i, j) -testMatrix.get(i, j)));
					equal=false;
					count_unequal++;
					difference = difference + Math.abs(validMatrix.get(i, j) -testMatrix.get(i, j));
					if (parameterwithoutWV>=j) {
						System.out.println("row: "+i+" column: "+j+":"+validMatrix.get(i, j)+" vs "+testMatrix.get(i, j));
					}
					
					//incorrect = incorrect + "("+i+","+j+") ";
				}else{
					//correct = correct + "("+i+","+j+") ";
				}
			}
		}
		if (equal==false) {
			//System.out.println("correct: "+correct);
			//System.out.println("incorrect: "+incorrect);
			System.out.println("Unequal values: "+count_unequal+ " of " +validMatrix.size() +" with total abs difference of "+ difference +" and normalized by unequals: "+ difference/count_unequal + "in relation to whole matrix:" + (difference/(validMatrix.zSum()/100)));
		}
		
		return equal;
	}
	public DoubleMatrix2D correctColtMatrixes2D(DoubleMatrix2D validMatrix, DoubleMatrix2D testMatrix){
		DoubleMatrix2D correctedmatrix = new DenseDoubleMatrix2D(validMatrix.rows(),validMatrix.columns()).assign(testMatrix);
		for (int i = 0; i < validMatrix.rows(); i++) {
			for (int j = 0; j < validMatrix.columns(); j++) {
				if (validMatrix.get(i, j) != testMatrix.get(i, j)) {
						correctedmatrix.set(i, j, validMatrix.get(i, j));
						//System.out.println("row: "+i+" column: "+j+":"+one.get(i, j)+" vs "+two.get(i, j));
				}
			}
		}
		
		return correctedmatrix;
	}
	private void stackFromNumpyToParametersP(String path) throws IOException{
		System.out.println("stack from numpy to parameters");
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    double[] theta = new double[(int)theta_initalP.size()];
	    //double[] theta = new double[(embeddingSize*embeddingSize*sliceSize+sliceSize+sliceSize+2*embeddingSize*sliceSize)*numberOfRelations+embeddingSize*numOfWords];
	    int count = 0;
	    while (line != null) {
	    	/*String[] linesplit = line.split(",");
	    	System.out.println("linesplit size: " +linesplit.length);
	    	for (int i = 0; i < linesplit.length; i++) {
	    		theta[count] = Double.parseDouble(linesplit[i]);
		    	count++;
			}*/
	    	theta[count] = Double.parseDouble(line);
	    	count++;
	    	line = br.readLine();
	    }
	    System.out.println("Counter: "+count+ "vs " + theta_initalP.size()+"");
		System.out.println("stackFromNumpyToParametersP");
		//System.out.println("Theta_input size"+theta.length());
		//Read the configuration from concatenate flattened vector to the specific paramters: w,v,b,u,...
		int readposition = 0;
		int readposition2 = 0;
		//int w_size = embeddingSize*embeddingSize*sliceSize; // number of values for paramter w for one relation
		int w_size = embeddingSize*embeddingSize;
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		int wordvectors_size = embeddingSize*numOfWords;
		
		DoubleMatrix1D thetaP = new DenseDoubleMatrix1D(theta.length).assign(theta);
		//load w:
		for (int r = 0; r < wP.size(); r++) {
			wP.get(r).fillLikeNumpy(thetaP.viewPart(readposition2, w_size*sliceSize));
			readposition2 = readposition2+w_size*sliceSize;
			for (int j = 0; j < sliceSize; j++) {
				readposition = readposition+w_size;
			}

		}
		System.out.println("wP.get(0).getSlice(0).viewPart(0,0,5, 5): "+wP.get(0).getSlice(0).viewPart(0,0,5, 5));
		//load v:
		for (int r = 0; r < vP.size(); r++) {
			vP.put(r,thetaP.viewPart(readposition, v_size).reshape( sliceSize,2* embeddingSize).viewDice());
			readposition = readposition+v_size;
		}
		System.out.println("vP.get(1).viewPart(0,0,3, 3): "+vP.get(1).viewPart(0,0,3, 3));
		//load b:
		for (int r = 0; r < bP.size(); r++) {
			bP.put(r,thetaP.viewPart(readposition, b_size).reshape(1, sliceSize));
			readposition = readposition+b_size;
			//System.out.println("bP: "+bP);

		}

		//load u:
		for (int r = 0; r < uP.size(); r++) {
			uP.put(r,thetaP.viewPart(readposition, u_size).reshape(sliceSize,1));
			readposition = readposition+u_size;
			//System.out.println("uP: "+uP);
		}
		//load word vectors:
		System.out.println("thetaP.viewPart: "+thetaP.viewPart(readposition,5));
		wordvectorsP.assign(thetaP.viewPart(readposition, wordvectors_size).reshape(numOfWords,embeddingSize).viewDice());
	}
	
	public IPair<Double, double[]> costfunction(double[] _theta){
		// Load input paramter(theta) into corresponding INDArray variables for loss / cost computation
				//System.out.println("thetaP sum: "+new DenseDoubleMatrix1D(_theta).zSum());
				String started = ""+new Date().toString();
				stackToParametersP(new DenseDoubleMatrix1D(_theta));
				// Initialize entity vectors and their gradient as matrix of zeros
				DoubleMatrix2D entity_vectorsP = new DenseDoubleMatrix2D(embeddingSize, numberOfEntities).assign(0);
				DoubleMatrix2D entity_vectors_gradP = new DenseDoubleMatrix2D(embeddingSize, numberOfEntities).assign(0);
				update=0;
				entity_vectorsP = tbj.createVectorsForEachEntityByWordVectorsWithPreloadIdicesP(wordvectorsP);
				
				// Initialize cost as zero
				double costP = 0;
				
				// Use hashmaps to store parameter gradients for each relation
				HashMap<Integer, TensorP> w_gradP = new HashMap<Integer, TensorP>();
				HashMap<Integer, DoubleMatrix2D> v_gradP = new HashMap<Integer, DoubleMatrix2D>();
				HashMap<Integer, DoubleMatrix2D> u_gradP = new HashMap<Integer, DoubleMatrix2D>();
				HashMap<Integer, DoubleMatrix2D> b_gradP = new HashMap<Integer, DoubleMatrix2D>();
				
				for (int r = 0; r < numberOfRelations; r++) {			
					// Get a list of examples / tripples for the ith relation
					ArrayList<Tripple> tripplesOfRelationR = tbj.getBatchJobTripplesOfRelation(r);
					//System.out.println(tripplesOfRelationR.size()+" Trainingsexample for relation r="+r+"of "+numberOfRelations);
					//Are there training tripples availabe (if the batchjob size is very small, not for each realation is tripple available
					if (tripplesOfRelationR.size() != 0) {
						//System.out.println("compute gradients, there are training data availabe...");
					
						// Initialize entity and rel index lists				
						DenseDoubleMatrix1D e1P = new DenseDoubleMatrix1D(tbj.getEntitiy1IndexNumbers(tripplesOfRelationR).transpose().data().asDouble());
						DenseDoubleMatrix1D e2P = new DenseDoubleMatrix1D(tbj.getEntitiy2IndexNumbers(tripplesOfRelationR).transpose().data().asDouble());
						DenseDoubleMatrix1D e3P = new DenseDoubleMatrix1D(tbj.getEntitiy3IndexNumbers(tripplesOfRelationR).transpose().data().asDouble());
						//System.out.println("e1P: "+e1P.viewPart(0, 35));
						// Initialize entity vector lists with zeros
						DenseDoubleMatrix2D entityVectors_e1P = new DenseDoubleMatrix2D(embeddingSize,tripplesOfRelationR.size());
						DenseDoubleMatrix2D entityVectors_e2P = new DenseDoubleMatrix2D(embeddingSize,tripplesOfRelationR.size());
						DenseDoubleMatrix2D entityVectors_e3P = new DenseDoubleMatrix2D(embeddingSize,tripplesOfRelationR.size());
						DenseDoubleMatrix2D entityVectors_neg_e1P = new DenseDoubleMatrix2D(embeddingSize,tripplesOfRelationR.size());
						DenseDoubleMatrix2D entityVectors_neg_e2P = new DenseDoubleMatrix2D(embeddingSize,tripplesOfRelationR.size());
						DenseDoubleMatrix1D e1_negP = new DenseDoubleMatrix1D(tripplesOfRelationR.size());
						DenseDoubleMatrix1D e2_negP = new DenseDoubleMatrix1D(tripplesOfRelationR.size());
						
						// Get only entity vectors of training examples of the this / rth relation
						//System.out.println("numberOfEntities: "+numberOfEntities);
						for (int j = 0; j < tripplesOfRelationR.size(); j++) {
							Tripple tripple = tripplesOfRelationR.get(j);
							entityVectors_e1P.viewColumn(j).assign(entity_vectorsP.viewColumn(tripple.getIndex_entity1()));
							entityVectors_e2P.viewColumn(j).assign(entity_vectorsP.viewColumn(tripple.getIndex_entity2()));
							entityVectors_e3P.viewColumn(j).assign(entity_vectorsP.viewColumn(tripple.getIndex_entity3_corrupt()));
						}
						//System.out.println("entityVectors_e1P: "+entityVectors_e1P.viewPart(0, 0, 5, 35));
						// Choose entity vectors for negative training example based on random
						if (tbj.getRandom_training_combination_handler()>0.5) {
							entityVectors_neg_e1P.assign(entityVectors_e1P);
							entityVectors_neg_e2P.assign(entityVectors_e3P);
							e1_negP.assign(e1P);
							e2_negP.assign(e3P);
						}else{
							entityVectors_neg_e1P.assign(entityVectors_e3P);
							entityVectors_neg_e2P.assign(entityVectors_e2P);
							e1_negP.assign(e3P);
							e2_negP.assign(e2P);
						}
						// Initialize pre-activations of the tensor network as matrix of zeros
				
						// Add contribution of W
						DenseDoubleMatrix2D sliceP = new DenseDoubleMatrix2D(embeddingSize,embeddingSize);

						DoubleMatrix2D preactivation_posP = new DenseDoubleMatrix2D(sliceSize, tripplesOfRelationR.size()).assign(0);
						DoubleMatrix2D preactivation_negP = new DenseDoubleMatrix2D(sliceSize, tripplesOfRelationR.size()).assign(0);
						//DoubleMatrix2D productMatrix = algebra.mult(entityVectors_e1P, d);
						for (int slice = 0; slice < sliceSize; slice++) {
							sliceP.assign(wP.get(r).getSlice(slice));
							DoubleMatrix2D temp = entityVectors_e1P.copy();
							DoubleMatrix2D temp_neg = entityVectors_neg_e1P.copy();
							temp.assign(sliceP.zMult(entityVectors_e2P, null),DoubleFunctions.mult);
							temp_neg.assign(sliceP.zMult(entityVectors_neg_e2P, null),DoubleFunctions.mult);
							for (int i = 0; i < entityVectors_e1P.columns(); i++) {
								preactivation_posP.set(slice, i, temp.viewColumn(i).zSum());
								preactivation_negP.set(slice, i, temp_neg.viewColumn(i).zSum());
							}
						}
						
						// Add contribution of V / W2
						DoubleMatrix2D vOfThisRelation_TP = vP.get(r).viewDice();
						DoubleFactory2D F = DoubleFactory2D.dense;
						DoubleMatrix2D vstack_pos1_P = F.appendRows(entityVectors_e1P, entityVectors_e2P);
						DoubleMatrix2D vstack_neg1_P = F.appendRows(entityVectors_neg_e1P, entityVectors_neg_e2P);
						// Add contribution of bias b
						DoubleMatrix2D bOfThisRelation_TP = bP.get(r).viewDice();

						DoubleMatrix2D res_pos = vOfThisRelation_TP.zMult(vstack_pos1_P, null);
						DoubleMatrix2D res_neg = vOfThisRelation_TP.zMult(vstack_neg1_P, null);
						for (int i = 0; i < entityVectors_e1P.columns(); i++) {
							res_pos.viewColumn(i).assign(bOfThisRelation_TP.viewColumn(0), DoubleFunctions.plus);
							res_neg.viewColumn(i).assign(bOfThisRelation_TP.viewColumn(0), DoubleFunctions.plus);
						}
						preactivation_posP.assign(res_pos, DoubleFunctions.plus);
						preactivation_negP.assign(res_neg, DoubleFunctions.plus);
						//System.out.println("preactivation_posP: "+preactivation_posP.viewPart(0, 0, 3, 35));
						//System.out.println("preactivation_negP: "+preactivation_negP.viewPart(0, 0, 3, 3));

						// Apply the activation function
						DoubleMatrix2D z_actviation_posP = new DenseDoubleMatrix2D(preactivation_posP.rows(), preactivation_posP.columns());
						DoubleMatrix2D z_actviation_negP = new DenseDoubleMatrix2D(preactivation_negP.rows(), preactivation_negP.columns());
						for (int row = 0; row < preactivation_posP.rows(); row++) {
							for (int col = 0; col < preactivation_posP.columns(); col++) {
								z_actviation_posP.set(row, col, Math.tanh(preactivation_posP.get(row, col)));
								z_actviation_negP.set(row, col, Math.tanh(preactivation_negP.get(row, col)));
							}
						}
						
						// Calculate scores for positive and negative examples
						DoubleMatrix2D score_posP = uP.get(r).viewDice().zMult(z_actviation_posP,null);
						DoubleMatrix2D score_negP = uP.get(r).viewDice().zMult(z_actviation_negP,null);
						//System.out.println(score_posP.viewPart(0, 0, 1, 35));
						//System.out.println(score_negP.viewPart(0, 0, 1, 3));
						//Filter for training examples, (that already predicted correct and dont need to be in account for further optimization of the paramters)			
						// https://groups.google.com/forum/?hl=en#!topic/recursive-deep-learning/chDXI1S2RHU
						DoubleMatrix1D indxP = new DenseDoubleMatrix1D(score_negP.columns());
						for (int i = 0; i < score_negP.columns(); i++) {
							//System.out.println("score pos: " + (score_posP.get(0, i)+1) +" > "+score_negP.get(0, i));
							if ((score_posP.get(0, i)+1) > score_negP.get(0, i)) {
								indxP.set(i, 1);
								//System.out.println(i + ": score pos: " + (score_posP.get(0, i)+1) +" > "+score_negP.get(0, i));
							}else{
								indxP.set(i, 0);
							}		
						}
						//System.out.println("score_posP: "+score_posP.zSum() + " | score_negP: "+score_negP.zSum());
						//System.out.println("score_posP: "+score_posP.rows()+" | "+score_posP.columns() + " | score_negP: "+score_negP.rows() +" | "+score_negP.columns());
						//System.out.println("indxP: " +indxP.viewPart(0, 20));
						//System.out.println("theta2 sum: "+parametersToStackP(wP, vP, bP, uP, wordvectorsP).zSum());
						
						costP = costP + score_posP.assign(score_negP, DoubleFunctions.minus).assign(new DenseDoubleMatrix2D(score_posP.rows(),score_posP.columns()).assign(1), DoubleFunctions.plus).zMult(indxP, null).zSum();
						//System.out.println("cost: "+costP);
						//Initialize W and V gradients as matrix of zero			
						w_gradP.put(r, new TensorP(embeddingSize, embeddingSize, sliceSize, 0));
						v_gradP.put(r, new DenseDoubleMatrix2D(2*embeddingSize,sliceSize).assign(0));
						
						//Number of examples contributing to error
						int numOfWrongExamplesP = (int)indxP.zSum();
						//System.out.println("numOfWrong: "+numOfWrongExamplesP);
						update = update + numOfWrongExamplesP;
						// For filtering matrixes, first get array with columns where indx is 1
						DoubleMatrix2D entVecE1RelP_fil = new DenseDoubleMatrix2D(embeddingSize, numOfWrongExamplesP);
						DoubleMatrix2D entVecE2RelP_fil = new DenseDoubleMatrix2D(embeddingSize, numOfWrongExamplesP);
						DoubleMatrix2D entVecE1Rel_negP_fil = new DenseDoubleMatrix2D(embeddingSize, numOfWrongExamplesP);
						DoubleMatrix2D entVecE2Rel_negP_fil = new DenseDoubleMatrix2D(embeddingSize, numOfWrongExamplesP);
						DoubleMatrix2D z_actviation_posP_fil = new DenseDoubleMatrix2D(sliceSize, numOfWrongExamplesP);
						DoubleMatrix2D z_actviation_negP_fil = new DenseDoubleMatrix2D(sliceSize, numOfWrongExamplesP);
						IntMatrix1D e1_filteredP = new DenseIntMatrix1D(numOfWrongExamplesP);
						IntMatrix1D e2_filteredP = new DenseIntMatrix1D(numOfWrongExamplesP);
						IntMatrix1D e1_neg_filteredP = new DenseIntMatrix1D(numOfWrongExamplesP);
						IntMatrix1D e2_neg_filteredP = new DenseIntMatrix1D(numOfWrongExamplesP);
						int counterP=0;
						for (int i = 0; i < indxP.size(); i++) {	
							if (indxP.getQuick(i)==1) {
								entVecE1RelP_fil.viewColumn(counterP).assign(entityVectors_e1P.viewColumn(i));
								entVecE2RelP_fil.viewColumn(counterP).assign(entityVectors_e2P.viewColumn(i));
								entVecE1Rel_negP_fil.viewColumn(counterP).assign(entityVectors_neg_e1P.viewColumn(i));
								entVecE2Rel_negP_fil.viewColumn(counterP).assign(entityVectors_neg_e2P.viewColumn(i));
								z_actviation_posP_fil.viewColumn(counterP).assign(z_actviation_posP.viewColumn(i));
								z_actviation_negP_fil.viewColumn(counterP).assign(z_actviation_negP.viewColumn(i));
								e1_filteredP.set(counterP, (int)e1P.getQuick(i));
								e2_filteredP.set(counterP, (int)e2P.getQuick(i));
								e1_neg_filteredP.set(counterP, (int)e1_negP.getQuick(i));
								e2_neg_filteredP.set(counterP, (int)e2_negP.getQuick(i));
								counterP++;
							}
						}
						// Calculate U[i] gradient
						DoubleMatrix2D u_gradPMatrix = new DenseDoubleMatrix2D(sliceSize,1);
						DoubleMatrix2D tempmatrixnew = new DenseDoubleMatrix2D(z_actviation_posP_fil.rows(),z_actviation_posP_fil.columns()).assign(z_actviation_posP_fil);
						for (int i = 0; i < sliceSize; i++) {
							u_gradPMatrix.set(i, 0, tempmatrixnew.viewRow(i).assign(z_actviation_negP_fil.viewRow(i), DoubleFunctions.minus).zSum());
						}
						u_gradP.put(r, u_gradPMatrix);
							
						//Calculate U * f'(z) terms useful for other gradient calculation
						DoubleMatrix2D diff_pos=  new DenseDoubleMatrix2D(z_actviation_posP_fil.rows(),z_actviation_posP_fil.columns()).assign(1).assign(z_actviation_posP_fil.assign(DoubleFunctions.pow(2)), DoubleFunctions.minus);
						DoubleMatrix2D diff_neg=  new DenseDoubleMatrix2D(z_actviation_negP_fil.rows(),z_actviation_negP_fil.columns()).assign(1).assign(z_actviation_negP_fil.assign(DoubleFunctions.pow(2)), DoubleFunctions.minus);
						for (int i = 0; i < diff_pos.columns(); i++) {
							diff_pos.viewColumn(i).assign(uP.get(r).viewColumn(0),DoubleFunctions.mult);
							diff_neg.viewColumn(i).assign(uP.get(r).viewColumn(0),DoubleFunctions.mult).assign(DoubleFunctions.neg);
						}
						
						// Calculate 'b[i]' gradient
						DoubleMatrix2D diff_pos2 =  new DenseDoubleMatrix2D(diff_pos.rows(), diff_pos.columns()).assign(diff_pos);
						DoubleMatrix2D bgradP = new DenseDoubleMatrix2D(1,sliceSize).assign(0);
						for (int i = 0; i < sliceSize; i++) {
							bgradP.set(0, i, diff_pos2.viewRow(i).assign(diff_neg.viewRow(i), DoubleFunctions.plus).zSum());
						}
						b_gradP.put(r, bgradP);
									
						// Values necessary for further compressed sparse row computations (not implemented in ND4j, with dense matrix: bad performance)
						INDArray values = Nd4j.ones(numOfWrongExamplesP);
						INDArray rows2 = Nd4j.arange(0, numOfWrongExamplesP);
						
						SparseRCDoubleMatrix2D e1_sparse = null;
						SparseRCDoubleMatrix2D e2_sparse= null;
						SparseRCDoubleMatrix2D e1_neg_sparse= null;
						SparseRCDoubleMatrix2D e2_neg_sparse= null;
						
						if (numOfWrongExamplesP!=0) {
							e1_sparse = new SparseRCDoubleMatrix2D(numOfWrongExamplesP,numberOfEntities, INDArrayToIntArray(rows2), e1_filteredP.toArray(),1.0,false,false);
							e2_sparse = new SparseRCDoubleMatrix2D(numOfWrongExamplesP,numberOfEntities, INDArrayToIntArray(rows2), e2_filteredP.toArray(),1.0,false,false);
							e1_neg_sparse = new SparseRCDoubleMatrix2D(numOfWrongExamplesP,numberOfEntities, INDArrayToIntArray(rows2), e1_neg_filteredP.toArray(),1.0,false,false);
							e2_neg_sparse = new SparseRCDoubleMatrix2D(numOfWrongExamplesP,numberOfEntities, INDArrayToIntArray(rows2), e2_neg_filteredP.toArray(),1.0,false,false);
						}
						//System.out.println("theta3 sum: "+parametersToStackP(wP, vP, bP, uP, wordvectorsP).zSum());
						//Initialize w gradient for this relation
						if (tripplesOfRelationR.size()!=0 & numOfWrongExamplesP!=0) {			
							for (int k = 0; k < sliceSize; k++) {				
								// U * f'(z) values corresponding to one slice
								DoubleMatrix1D temp_posP = diff_pos.viewRow(k);
								DoubleMatrix1D temp_negP = diff_neg.viewRow(k);
								
								//Calculate 'k'th slice of 'W[i]' gradient
								DoubleMatrix2D entVecE1RelP2= new DenseDoubleMatrix2D(entVecE1RelP_fil.rows(),entVecE1RelP_fil.columns()).assign(entVecE1RelP_fil);
								DoubleMatrix2D entVecE1Rel_neg_P2= new DenseDoubleMatrix2D(entVecE1Rel_negP_fil.rows(),entVecE1Rel_negP_fil.columns()).assign(entVecE1Rel_negP_fil);
								for (int i = 0; i < entVecE1RelP_fil.rows(); i++) {
									entVecE1RelP2.viewRow(i).assign(temp_posP, DoubleFunctions.mult);
									entVecE1Rel_neg_P2.viewRow(i).assign(temp_negP, DoubleFunctions.mult);
								}
								
								DoubleMatrix2D wgradres1 = entVecE1RelP2.zMult(entVecE2RelP_fil.viewDice(), null);
								DoubleMatrix2D wgradres1_neg = entVecE1Rel_neg_P2.zMult(entVecE2Rel_negP_fil.viewDice(), null);
								wgradres1.assign(wgradres1_neg, DoubleFunctions.plus);
								//System.out.println("w_grad: "+wgradres1.viewPart(0, 0, 5, 5));
								w_gradP.get(r).setSlice(k, wgradres1);
								
								//Calculate 'k'th slice of V gradient	
								DoubleMatrix2D sum_vP = new DenseDoubleMatrix2D(2*embeddingSize,1);
								DoubleMatrix2D vstack_pos_P = F.appendRows(entVecE1RelP_fil, entVecE2RelP_fil);
								DoubleMatrix2D vstack_neg_P = F.appendRows(entVecE1Rel_negP_fil, entVecE2Rel_negP_fil);
								for (int i = 0; i < vstack_pos_P.rows(); i++) {
									sum_vP.viewRow(i).assign(vstack_pos_P.viewRow(i).assign(temp_posP, DoubleFunctions.mult).assign(vstack_neg_P.viewRow(i).assign(temp_negP, DoubleFunctions.mult), DoubleFunctions.plus).zSum());
								}
								v_gradP.get(r).viewColumn(k).assign(sum_vP.vectorize());
								
								// Add contribution of V term in the entity vectors' gradient	
								DoubleMatrix2D kthSliceOfVP = vP.get(r).viewColumn(k).reshape(embeddingSize*2, 1);
								DoubleMatrix2D vposP = kthSliceOfVP.zMult(temp_posP.reshape(1, (int)temp_posP.size()),null);
								DoubleMatrix2D vnegP = kthSliceOfVP.zMult(temp_negP.reshape(1, (int)temp_negP.size()),null);
								
								DoubleMatrix2D v1P = vposP.viewPart(0,0 , embeddingSize, vposP.columns()).zMult(e1_sparse, null);
								DoubleMatrix2D v2P = vposP.viewPart(embeddingSize,0 , embeddingSize, vposP.columns()).zMult(e2_sparse, null);
								DoubleMatrix2D v3P = vnegP.viewPart(0,0 , embeddingSize, vnegP.columns()).zMult(e1_neg_sparse, null);
								DoubleMatrix2D v4P = vnegP.viewPart(embeddingSize,0 , embeddingSize, vnegP.columns()).zMult(e2_neg_sparse, null);
								entity_vectors_gradP.assign(v1P, DoubleFunctions.plus).assign(v2P, DoubleFunctions.plus).assign(v3P, DoubleFunctions.plus).assign(v4P, DoubleFunctions.plus);

								// Add contribution of 'W[i]' term in the entity vectors' gradient
								DoubleMatrix2D sliceWP= new DenseDoubleMatrix2D(embeddingSize,embeddingSize);
								DoubleMatrix2D sliceWTP= new DenseDoubleMatrix2D(embeddingSize,embeddingSize);
								sliceWP.assign(wP.get(r).getSlice(k));
								sliceWTP.assign(sliceWP.viewDice());
								//
								DoubleMatrix2D w1P = sliceWP.zMult(entVecE2RelP_fil, null);
								DoubleMatrix2D w2P = sliceWTP.zMult(entVecE1RelP_fil, null);
								DoubleMatrix2D w3P = sliceWP.zMult(entVecE2Rel_negP_fil, null);
								DoubleMatrix2D w4P = sliceWTP.zMult(entVecE1Rel_negP_fil, null);
								
								for (int i = 0; i < w1P.rows(); i++) {
									w1P.viewRow(i).assign(temp_posP, DoubleFunctions.mult);
									w2P.viewRow(i).assign(temp_posP, DoubleFunctions.mult);
									w3P.viewRow(i).assign(temp_negP, DoubleFunctions.mult);
									w4P.viewRow(i).assign(temp_negP, DoubleFunctions.mult);
								}
								w1P= w1P.zMult(e1_sparse, null);
								w2P=w2P.zMult(e2_sparse, null);
								w3P=w3P.zMult(e1_neg_sparse, null);
								w4P=w4P.zMult(e2_neg_sparse, null);
								entity_vectors_gradP.assign(w1P, DoubleFunctions.plus).assign(w2P, DoubleFunctions.plus).assign(w3P, DoubleFunctions.plus).assign(w4P, DoubleFunctions.plus);
							}
						}else{
							// Filling with zeros if there is no training example for this relation in the training batch
							//System.out.println("no trainingsexample, gradients are zero, num of Wrong:" + numOfWrongExamplesP);			
						}
						
						// Normalize the gradients with the training batch size
						
						for (int i = 0; i < sliceSize; i++) {
							w_gradP.get(r).getSlice(i).assign(DoubleFunctions.div(batchSize));
						}
						//System.out.println("w_gradP sum: "+(w_gradP.get(r).getSlice(0).zSum()+w_gradP.get(r).getSlice(1).zSum()+w_gradP.get(r).getSlice(2).zSum()));
						//System.out.println("w_gradP sum: "+(w_gradP.get(r).getSlice(0).zSum()));
						v_gradP.get(r).assign(DoubleFunctions.div(batchSize));
						//System.out.println("v_gradP.get(r) sum: "+v_gradP.get(r).zSum());
						b_gradP.get(r).assign(DoubleFunctions.div(batchSize));
						//System.out.println("b_gradP.get(r) sum: "+b_gradP.get(r).zSum());
						u_gradP.get(r).assign(DoubleFunctions.div(batchSize));
						//System.out.println("u_gradP.get(r) sum: "+u_gradP.get(r).zSum());
					}else{
						//System.out.println("next relation, maybe there is a training tripple availabe...");		
						w_gradP.put(r, new TensorP(embeddingSize,embeddingSize,sliceSize,0));
						v_gradP.put(r, new DenseDoubleMatrix2D(2*embeddingSize,sliceSize));
						b_gradP.put(r, new DenseDoubleMatrix2D(1,sliceSize));
						u_gradP.put(r, new DenseDoubleMatrix2D(sliceSize,1));	
					}
					
				}
				// Initialize word vector gradients as a matrix of zeros
				DoubleMatrix2D wv_grad = new DenseDoubleMatrix2D(embeddingSize, numOfWords).assign(0);
				// Calculate word vector gradients from entity gradients
				for (int i = 0; i < numberOfEntities; i++) {
					int[] wordindexes = tbj.getWordIndexes(i);
					DoubleMatrix1D x = new DenseDoubleMatrix1D(embeddingSize).assign(wordindexes.length);
					DoubleMatrix1D y =entity_vectors_gradP.viewColumn(i).assign(x, DoubleFunctions.div);	
					for (int j = 0; j < wordindexes.length; j++) {
						wv_grad.viewColumn(wordindexes[j]).assign(y, DoubleFunctions.plus);
					}
					
				}
				//System.out.println("theta4 sum: "+parametersToStackP(wP, vP, bP, uP, wordvectorsP).zSum());
				// Normalize word vector gradients and cost by the training batch size
				wv_grad.assign(DoubleFunctions.div(batchSize));
				//System.out.println("wv_grad sum: "+wv_grad.zSum());
				costP = costP / batchSize;
				// Get stacked gradient vector and parameter vector	
				DoubleMatrix1D theta_gradP =parametersToStackP(w_gradP, v_gradP, b_gradP, u_gradP, wv_grad);

				//System.out.println("thetaP: sum "+new DenseDoubleMatrix1D((int)_theta.length).assign(_theta));
				DoubleMatrix1D thetaP =parametersToStackP(wP, vP, bP, uP,wordvectorsP);
				DoubleMatrix1D temp= new DenseDoubleMatrix1D((int)thetaP.size()).assign(thetaP);
				temp.assign(DoubleFunctions.pow(2));
				costP = costP + (0.5 * (lamda * temp.zSum()));
				thetaP.assign(DoubleFunctions.mult(lamda));
				theta_gradP.assign(thetaP,DoubleFunctions.plus);
				System.out.println("Cost: "+costP+ " | #Wrong: "+update+" | start: "+started+" |end: "+new Date().toString());
				//System.out.println("Cost: "+costP+" Theta grad sum:"+theta_gradP.zSum()+ " | start: "+started+" |end: "+new Date().toString());
				// IPair<Double, double[]>
				/*
				try {
					stackFromNumpyToParametersP("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet - Copy//_1_grad.txt");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					System.out.println("CATCHED???");
					e.printStackTrace();
				}
				thetaP =parametersToStackP(wP, vP, bP, uP, wordvectorsP);
				System.out.println(equalsColtMatrixes2D(thetaP.reshape(1, (int)thetaP.size()), theta_gradP.reshape(1, (int)theta_gradP.size())));
				//System.out.println("w_gradP.get(0).getSlice(0): "+w_gradP.get(0).getSlice(0).viewPart(0,0,10,10));	
				System.out.println("Numpy wP.get(0).getSlice(0): "+wP.get(0).getSlice(0).viewPart(0,0,10,10));	
				stackToParametersP(theta_gradP);
				*/
				 return BasicPair.make( costP, theta_gradP.toArray() );
	}

	public INDArray computeBestThresholds(DoubleMatrix1D _theta, ArrayList<Tripple> _devTrippels){
		//load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParametersP(_theta);
		// create entity vectors from word vectors
		//INDArray entity_vectors= this.getDatafactory().createVectorsForEachEntityByWordVectors();
		DoubleMatrix2D entity_vectors = this.getDatafactory().createVectorsForEachEntityByWordVectorsWithPreloadIdicesP(wordvectorsP);
		
		INDArray dev_scores = Nd4j.zeros(_devTrippels.size());
		
		for (int i = 0; i < _devTrippels.size(); i++) {
			//Get entity 1 and 2 for examples of ith relation
			double score = calculateScoreOfaTripple(_devTrippels.get(i), entity_vectors);
			dev_scores.putScalar(i, score);
			if (i<3) {
				System.out.println(_devTrippels.get(i).toString() + " | score: "+_devTrippels.get(i).getScore());
			}
		}
		
		// Maximum and Minimum of the scores
		INDArray score_min = Nd4j.min(dev_scores);
		INDArray score_max = Nd4j.max(dev_scores);
		System.out.println("score min: "+score_min);
		System.out.println("score max: "+score_max);
		
		// Initialize thereshold and accuracies
		INDArray best_theresholds = Nd4j.zeros(numberOfRelations,1);
		INDArray best_accuracies = Nd4j.zeros(numberOfRelations,1);
		
		for (int i = 0; i < numberOfRelations; i++) {
			best_theresholds.put(i, score_min);
			best_theresholds.putScalar(i, -1);
		}
		
		double score_temp = score_min.getDouble(0); // contains the value of the score that classifies a tripple as correct or incorrect
		double interval = 0.01; // the value that updates the score_temp to find a better thereshold for classification of correct or in correct
		
		//Check for the best accuracy at intervals betweeen 'score_min' and 'score_max'
		
		while (score_temp <= score_max.getDouble(0)) {
			//Check accuracy for the ith relation
			for (int i = 0; i < numberOfRelations; i++) {			
				ArrayList<Tripple> tripplesOfThisRelation = tbj.getTripplesOfRelation(i, _devTrippels);
				double temp_accuracy=0;
				
				//compare the score of each tripple with the label
				for (int j = 0; j < tripplesOfThisRelation.size(); j++) {
					//double scoreOfThisTripple = tripplesOfThisRelation.get(j).getScore();
					//System.out.println("if: "+tripplesOfThisRelation.get(j).getScore()+" <= "+score_temp);
					if (tripplesOfThisRelation.get(j).getScore() <= score_temp) {
						
						//Label of this tripple = 1;	//classification of this tripple as correct	
						if (tripplesOfThisRelation.get(j).getLabel() == 1) {
							temp_accuracy = temp_accuracy +1;
						}
					}else{
						//Label of this tripple = -1; //classification of this tripple as incorrect
						if (tripplesOfThisRelation.get(j).getLabel() == -1) {
							temp_accuracy = temp_accuracy +1;
						}
					}
					//if (scoreOfThisTripple == tripplesOfThisRelation.get(j).getLabel()) {
					//	temp_accuracy = temp_accuracy +1;
					//} 	
				}
				//System.out.println("temp_accuracy for "+i+" relation: "+temp_accuracy);
				temp_accuracy = temp_accuracy / new Double(tripplesOfThisRelation.size()); //current accuracy of prediction for this relation
				//System.out.println("temp_accuracy for "+i+" relation: "+temp_accuracy);
				
				//If the accuracy is better, update the threshold and accuracy values
				if (temp_accuracy > best_accuracies.getDouble(i)) {
					if (i==3) {
						//System.out.println("update temp_accuracy for "+i+" relation: "+temp_accuracy);
						//System.out.println("update score_temp for "+i+" relation: "+score_temp);
					}
					best_accuracies.putScalar(i, temp_accuracy);
					best_theresholds.putScalar(i, score_temp);
				}
				
			}
			score_temp = score_temp + interval;
		}
		// Print statistics:
		for (int i = 0; i < numberOfRelations; i++) {
			System.out.println("relation: "+i+" best acc: "+ best_accuracies.getDouble(i)+" with score threshold: "+best_theresholds.getDouble(i));
		}
		System.out.println("Accuracy of dev predictions: " + Nd4j.mean(best_accuracies));
		//return the best theresholds for the prediction
		return best_theresholds;
		
		
	}

	public INDArray getPrediction(DoubleMatrix1D _theta, ArrayList<Tripple> _testTripples, INDArray _bestThresholds) throws FileNotFoundException, UnsupportedEncodingException{
		//TSNE PrintWriter writer_vec = new PrintWriter("C://Users//Patrick//Documents//master arbeit//original_code//data//tsne//_mul_ent_vec.txt", "Cp1252");
		//TSNE PrintWriter writer_ent = new PrintWriter("C://Users//Patrick//Documents//master arbeit//original_code//data//tsne//_mul_uk_ent.txt", "Cp1252");
		ArrayList<String> vocab = new ArrayList<String>();
		// load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParametersP(_theta);
		// create entity vectors from word vectors
		DoubleMatrix2D entity_vectors = this.getDatafactory().createVectorsForEachEntityByWordVectorsWithPreloadIdicesP(wordvectorsP);
		
		// initialize array to store the predictions of in- and correct tripples of the test data
		INDArray predictions = Nd4j.zeros(_testTripples.size());
		INDArray accuracy = Nd4j.zeros(_testTripples.size()); // if predcition == lable -> 1 else 0
		INDArray accuracyForEachRelation = Nd4j.zeros(_testTripples.size(),2); // first column acc, second rel
		System.out.println("_testTripples.size(): "+_testTripples.size());
		int counter_correct=0;
		int counter_wrong=0;
		for (int i = 0; i < _testTripples.size(); i++) {
			//TSNE double score = calculateScoreOfaTripplePrint(_testTripples.get(i), entity_vectors,writer_vec,writer_ent,_bestThresholds,vocab);
			double score = calculateScoreOfaTripple(_testTripples.get(i),entity_vectors);
			//System.out.println("get score:" + _testTripples.get(i).getScore());
			if (_testTripples.get(i).getEntity1().contains("calciatore")|_testTripples.get(i).getEntity2().contains("calciatore") |_testTripples.get(i).getEntity2().contains("giocatore_di_lacrosse")) {
				System.out.println(_testTripples.get(i)+": "+score+" threshold: "+_bestThresholds.getDouble(_testTripples.get(i).getIndex_relation())+" (smaller as threshold means predicted as correct");
			}
			// calculate prediction based on previously calculate thersholds
			if(score<= _bestThresholds.getDouble(_testTripples.get(i).getIndex_relation())){
				//tripple is predicted as correct
				_testTripples.get(i).setPrediction(1);
				predictions.putScalar(i, 1);
				//compare tripple prediction with label
				if (_testTripples.get(i).getLabel()==1) {
					accuracy.putScalar(i, 1);
					accuracyForEachRelation.put(i,0,1);
					accuracyForEachRelation.put(i,1,_testTripples.get(i).getIndex_relation());
				}else{
					accuracy.putScalar(i, 0);
					accuracyForEachRelation.put(i,0,0);
					accuracyForEachRelation.put(i,1,_testTripples.get(i).getIndex_relation());
				}
			}else{
				//tripple is predicted as incorrect
				_testTripples.get(i).setPrediction(-1);
				predictions.putScalar(i, -1);
				//compare tripple prediction with label
				if (_testTripples.get(i).getLabel()==-1) {
					accuracy.putScalar(i, 1);
					accuracyForEachRelation.put(i,0,1);
					accuracyForEachRelation.put(i,1,_testTripples.get(i).getIndex_relation());
				}else{
					accuracy.putScalar(i, 0);
					accuracyForEachRelation.put(i,0,0);
					accuracyForEachRelation.put(i,1,_testTripples.get(i).getIndex_relation());
				}
			}
				
		}
		for (int i = 0; i < numberOfRelations; i++) {
			double count = 0;
			double correct = 0;
			for (int j = 0; j < accuracyForEachRelation.rows(); j++) {
				//System.out.println(j+": "+accuracyForEachRelation.getDouble(j, 1) );
				if (accuracyForEachRelation.getDouble(j, 1) == i) {
					count++;
					if (accuracyForEachRelation.getDouble(j, 0)== 1) {
						correct++;
					}
				}
			}
			System.out.println("count: "+count +" correct: "+correct);
			double acc = correct/count;
			System.out.println("Accuracy for relation "+i+": "+ acc);
		}
		System.out.println("Accuracy of predictions complete: " + Nd4j.mean(accuracy));
		//TSNE writer_vec.close();
		//TSNE writer_ent.close();
		return predictions; //array with predictions
		
	}

	private double calculateScoreOfaTripple(Tripple _tripple, DoubleMatrix2D _entity_vectors){
		//Get entity 1 and 2 for examples of ith relation
		DoubleMatrix1D entityVector1 = _entity_vectors.viewColumn(_tripple.getIndex_entity1());
		DoubleMatrix1D entityVector2 = _entity_vectors.viewColumn(_tripple.getIndex_entity2());

		int rel = _tripple.getIndex_relation();
		
		//DoubleMatrix2D vstacke1e2 = DoubleFactory2D.dense.(entityVector1.reshape(1, embeddingSize), entityVector2);
		DoubleMatrix1D vstacke1e2 = new DenseDoubleMatrix1D(2*embeddingSize);
		for (int i = 0; i < embeddingSize; i++) {
			vstacke1e2.set(i,entityVector1.get(i));
			vstacke1e2.set(i+100,entityVector2.get(i));
		}
		
		//DoubleMatrix2D vstack =  DoubleFactory2D.dense.appendRows(entityVector1, entityVector2);
		DoubleMatrix1D vstack = DoubleFactory1D.dense.append(entityVector1, entityVector2);
		
		// Calculate the prdediction score for the ith example
		double score_temp=0;

		for (int slice = 0; slice < sliceSize; slice++) {
			
			DoubleMatrix1D dotpro1 = wP.get(rel).getSlice(slice).zMult(entityVector2, null);
			double dotpro2d = entityVector1.zDotProduct(dotpro1);
			double dotpro3d = vP.get(rel).viewColumn(slice).zDotProduct(vstack);
			DoubleMatrix1D dotpro3 = new DenseDoubleMatrix1D(1);
			DoubleMatrix1D dotpro2 = new DenseDoubleMatrix1D(1);
			dotpro3.set(0, dotpro3d);
			dotpro2.set(0, dotpro2d);
			dotpro3.assign(bP.get(rel).viewColumn(slice), DoubleFunctions.plus);
			dotpro3.assign(dotpro2, DoubleFunctions.plus);
			//activation:
			dotpro3.assign(Math.tanh(dotpro3.get(0)));
			dotpro3.assign(uP.get(rel).viewRow(slice), DoubleFunctions.mult);
			
			score_temp = score_temp + dotpro3.get(0);
		}
		_tripple.setScore(score_temp);
		//System.out.println(_tripple.toString() + " | score: "+_tripple.getScore()+" | label: "+_tripple.getLabel());
		return score_temp;
	}

	private double calculateScoreOfaTripplePrint(Tripple _tripple, DoubleMatrix2D _entity_vectors,PrintWriter writer_vec,PrintWriter writer_ent, INDArray _bestThresholds, ArrayList<String> vocab) throws FileNotFoundException, UnsupportedEncodingException{
		//Get entity 1 and 2 for examples of ith relation
		
		DoubleMatrix1D entityVector1 = _entity_vectors.viewColumn(_tripple.getIndex_entity1());
		DoubleMatrix1D entityVector2 = _entity_vectors.viewColumn(_tripple.getIndex_entity2());
	
		int rel = _tripple.getIndex_relation();
		
		//DoubleMatrix2D vstacke1e2 = DoubleFactory2D.dense.(entityVector1.reshape(1, embeddingSize), entityVector2);
		DoubleMatrix1D vstacke1e2 = new DenseDoubleMatrix1D(2*embeddingSize);
		for (int i = 0; i < embeddingSize; i++) {
			vstacke1e2.set(i,entityVector1.get(i));
			vstacke1e2.set(i+100,entityVector2.get(i));
		}
		
		//DoubleMatrix2D vstack =  DoubleFactory2D.dense.appendRows(entityVector1, entityVector2);
		DoubleMatrix1D vstack = DoubleFactory1D.dense.append(entityVector1, entityVector2);
		
		// Calculate the prdediction score for the ith example
		double score_temp=0;
		String print_vec="";
		print_vec = ""+entityVector1.get(0);
		for (int i = 1; i < entityVector1.size(); i++) {
			print_vec = print_vec+","+entityVector1.get(i);
		}
		String print_vec2="";
		print_vec2 = ""+entityVector2.get(0);
		for (int i = 1; i < entityVector2.size(); i++) {
			print_vec2 = print_vec2+","+entityVector2.get(i);
		}

		for (int slice = 0; slice < sliceSize; slice++) {
			DoubleMatrix1D dotpro1 = wP.get(rel).getSlice(slice).zMult(entityVector2, null);
			double dotpro2d = entityVector1.zDotProduct(dotpro1);
			/*if (slice==0) {
				print_vec = ""+dotpro2d;
			}else{
				print_vec = print_vec+","+dotpro2d;
			}*/
			double dotpro3d = vP.get(rel).viewColumn(slice).zDotProduct(vstack);
			DoubleMatrix1D dotpro3 = new DenseDoubleMatrix1D(1);
			DoubleMatrix1D dotpro2 = new DenseDoubleMatrix1D(1);
			dotpro3.set(0, dotpro3d);
			dotpro2.set(0, dotpro2d);
			dotpro3.assign(bP.get(rel).viewColumn(slice), DoubleFunctions.plus);
			dotpro3.assign(dotpro2, DoubleFunctions.plus);
			
			//activation:
			dotpro3.assign(Math.tanh(dotpro3.get(0)));
			//exporting hidden layer representations
			/*if (slice==0) {
				print_vec = ""+dotpro3.get(0);
			}else{
				print_vec = print_vec+","+dotpro3.get(0);
			}*/
			dotpro3.assign(uP.get(rel).viewRow(slice), DoubleFunctions.mult);
			
			score_temp = score_temp + dotpro3.get(0);
		}
		if (rel==8) {
			if (!vocab.contains(_tripple.getEntity1())) {
				vocab.add(_tripple.getEntity1());
				writer_vec.println(print_vec);
				writer_ent.println(_tripple.getEntity1());
				//writer_ent.println(_tripple.getEntity1()+ "|"+(score_temp<=_bestThresholds.getDouble(rel))+"|"+_tripple.getLabel()+"|"+(""+score_temp).substring(0, 4));
			}
			if (!vocab.contains(_tripple.getEntity2())) {
				vocab.add(_tripple.getEntity2());
				writer_vec.println(print_vec2);
				writer_ent.println(_tripple.getEntity2());
				//writer_ent.println(_tripple.getEntity2()+ "|"+(score_temp<=_bestThresholds.getDouble(rel))+"|"+_tripple.getLabel()+"|"+(""+score_temp).substring(0, 4));
			}
			
			//writer_vec.println(print_vec);
			//writer_ent.println(_tripple.getEntity1()+"-"+_tripple.getEntity2()+ "|"+(score_temp<=_bestThresholds.getDouble(rel))+"|"+_tripple.getLabel()+"|"+(""+score_temp).substring(0, 4));
		}
		
		_tripple.setScore(score_temp);
		//System.out.println(_tripple.toString() + " | score: "+_tripple.getScore()+"/"+_bestThresholds.getDouble(rel)+" | label: "+_tripple.getLabel());
		return score_temp;
	}
	public void gradientchecking(double[] theta){
		System.out.println("Gradient Checking started");
		//costfunction returns cost and gradients
		IPair<Double, double[]> org = costfunction(theta);
		double[] theta_pos = new double[theta.length];
		double[] theta_neg = new double[theta.length];
		for (int i = 0; i < theta.length; i++) {
			theta_pos[i]= theta[i];
			theta_neg[i]=theta[i];
		}
		
		double mu = 1e-5;
		for (int k = 0; k < 20; k++) {
			theta_pos[k] = theta_pos[k] + mu;
			theta_neg[k] = theta_neg[k] - mu;
			IPair<Double, double[]> pos = costfunction(theta_pos);
			IPair<Double, double[]> neg = costfunction(theta_neg);
			System.out.println("Org: "+org.getSecond()[k] +" check:"+ ((pos.getSecond()[k]-neg.getSecond()[k])/(2*mu)));
			//System.out.println("Org: "+org.getSecond()[k] +"check:"+ ((pos.getSecond()[k]-neg.getSecond()[k])/(2*mu)));
			theta_pos[k] = theta_pos[k] - mu;
			theta_neg[k] = theta_neg[k] + mu;
		}
	}
} 
