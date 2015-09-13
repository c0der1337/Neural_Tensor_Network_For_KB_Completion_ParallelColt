

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import com.sun.xml.internal.bind.v2.model.core.ArrayInfo;

import edu.stanford.nlp.optimization.QNMinimizer;
import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.optimize.IOptimizer;
import edu.umass.nlp.optimize.LBFGSMinimizer;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.DoubleArrays;
import edu.umass.nlp.utils.IPair;
/**
 * Created by Patrick K. on 10/06/15.
 * A Neural Tensor Network for Knowledge Base Completion as described in the paper 
 * "Reasoning With Neural Tensor Networks For Knowledge Base Completion" 
 * (http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf 
 * by Richard Socher, Danqi Chen, Christopher D. Manning and Andrew Y. Ng. 
 * (http://www.socher.org/index.php/Main/ReasoningWithNeuralTensorNetworksForKnowledgeBaseCompletion)
 */
public class Run_NTN {

	public static void main(String[] args) throws IOException {
		Nd4j.dtype = DataBuffer.Type.DOUBLE;
		
		//Data path
		String data_path="";String theta_save_path=""; String theta_load_path="";
		
		//Training data
		String entities_file="entities.txt";
		String relations_file="relations.txt";
		String train_file="train.txt";
		String dev_file="dev.txt";
		String test_file="test.txt";
		String wordindices_file="wordindices.txt";
		String wordvecmatrix_file="wordvecmatrix.txt";
		
		//Paramters
		int batchSize = 200; 				// training batch size, org: 20000
		int numWVdimensions = 100; 			// size of the dimension of a word vector org: 100
		int numIterations = 500; 				// number of optimization iterations, every iteration with a new training batch job, org: 500
		int batch_iterations = 5;			// number of optimazation iterations for each batchs, org: 5
		int sliceSize = 4; 					// number of slices in the tensor w and v, org: 4
		int corrupt_size = 10; 				// corruption size, org: 10
		String activation_function= "tanh"; // [x] tanh or [] sigmoid, org:tanh
		double lamda = 1.0E-4;				// regularization parameter, org: 0.0001
		boolean optimizedLoad=false;		// only load word vectors that are neede for entity vectors (>50% less), org: false
		boolean reducedNumOfRelations = false; // reduce the number of relations to the first or first and second
		boolean minimizer = true;			// UMAS Minimizer = true, Stanford NLP Core QN Minimizer = false
		boolean loadOwnWE = true;			// if true, a own created word embedding space is loaded for training. If false: Sochers et al. word embeddings are loaded	
		boolean train_both = false;
		
		try {
			System.out.println("param0: thata save path: "+ args[0]);
			theta_save_path = args[0];
			System.out.println("param1: data path: "+ args[1]);
			data_path = args[1];
			System.out.println("param2: entities: "+ args[2]);
			entities_file= args[2];
			System.out.println("param3: relation: "+ args[3]);
			relations_file= args[3];
			System.out.println("param4: training data: "+ args[4]);
			train_file= args[4];
			System.out.println("param5: dev data: "+ args[5]);
			dev_file= args[5];
			System.out.println("param6: test data: "+ args[6]);
			test_file= args[6];
			System.out.println("param7: wordindices: "+ args[7]);
			wordindices_file= args[7];
			System.out.println("param8: word embeddings: "+ args[8]);
			wordvecmatrix_file= args[8];
			System.out.println("param9: slice: "+ args[9]);
			sliceSize = Integer.parseInt(args[9]);
			System.out.println("param10: batch size: "+ args[10]);
			batchSize = Integer.parseInt(args[10]);
			System.out.println("param11: training iterations: "+ args[11]);
			numIterations = Integer.parseInt(args[11]);
			System.out.println("param12: batch iterations: "+ args[12]);
			batch_iterations = Integer.parseInt(args[12]);
			System.out.println("param13: corrupt size:"+ args[13]);
			corrupt_size = Integer.parseInt(args[13]);
			
		} catch (Exception e) {
			System.out.println("+++ Standard Parameters used +++");
			data_path = "C://Users//Patrick//Documents//master arbeit//original_code//data//WordNet_Multi_UKDEIT//";
			theta_save_path = "C://Users//Patrick//Documents//master arbeit//";
			entities_file= "entities_mulUKDEIT.txt";
			relations_file= "relations.txt";
			train_file= "train_mulUKDEIT.txt";
			dev_file= "..//WordnetIT//org//devIT.txt";
			test_file= "..//WordnetIT//org//testIT.txt";
			wordindices_file= "wordindices_mulUKDEIT.txt";
			wordvecmatrix_file= "wordvecmatrixUKDEIT.txt";
			//Load pretrained weights:
			//theta_load_path = "C://Users//Patrick//Documents//master arbeit//weights//python//4Slice//mul_UKDEIT//theta_075_mulUKDEIT_MIX2.out";
		}

		System.out.println("NTN: batchSize: "+batchSize+" | SliceSize: "+sliceSize+" | numIterations:"+numIterations+" | corrupt_size: "+corrupt_size+"| activation func: "+ activation_function);
		
		//support utilities
		Util u = new Util();
		
		//Load data entities, relation, traingsdata, word vectors ...
		DataFactory df = DataFactory.getInstance(batchSize, corrupt_size, numWVdimensions, reducedNumOfRelations,train_both);
		df.loadEntitiesFromSocherFile(data_path + entities_file);
		df.loadRelationsFromSocherFile(data_path + relations_file);	
		df.loadTrainingDataTripplesE1rE2(data_path + train_file);
		df.loadDevDataTripplesE1rE2Label(data_path + dev_file);
		df.loadTestDataTripplesE1rE2Label(data_path + test_file);
		df.loadWordIndicesFromFile(data_path + wordindices_file);
		if (loadOwnWE) {
			df.loadWordVectorsFromFile(data_path + wordvecmatrix_file);
		}else{
			df.loadWordVectorsFromMatFile(data_path + "initEmbed.mat",optimizedLoad);
		}
		
		// Create the NTN and set the parameters for the NTN
		NTN t = new NTN(numWVdimensions, df.getNumOfentities(), df.getNumOfRelations(), df.getNumOfWords(), batchSize, sliceSize, activation_function, df, lamda, theta_load_path);
		t.connectDatafactory(df);
		
		//Load initialized parameters from file or via random initialization
		double[] theta = t.getTheta_inital();
		
		//gradient check
		
		//Load previous stored weights (possibility to load from python implementation weights)
		//double[] theta = Nd4j.readTxt("C://Users//Patrick//Documents//master arbeit//weights//python//4Slice//DE//theta_499DE.out", ",").data().asDouble();
		
		//Train	
		
		//System.out.println("UMAS: min iters: 0 " +"|tol: 1.0e-5 | max history: 10 | random word vectors");
		for (int i = 0; i < numIterations; i++) { 
			//Create a training batch by picking up (random) samples from training data	
			df.generateNewTrainingBatchJob();
			
			if (minimizer == true) {
				//Initilize optimizer and set optimizer options to 5 iterations
				LBFGSMinimizer.Opts optimizerOpts = new LBFGSMinimizer.Opts();				
				optimizerOpts.maxIters=batch_iterations;
				//from scipy L-BFGS-B:
				optimizerOpts.minIters=0;
				//optimizerOpts.tol=1.0e-5;
				//optimizerOpts.maxHistorySize=10;
				//Optimize the network using the training batch
				IOptimizer.Result res = (new LBFGSMinimizer()).minimize(t, theta, optimizerOpts);
				//System.out.println("res: "+res.didConverge + "| "+res.minObjVal);
				theta = res.minArg;
			}else{
				QNMinimizer qn = new QNMinimizer(10,true) ;
			    qn.terminateOnMaxItr(batch_iterations);
			    //qn.setM(10);
				theta = qn.minimize(t, 1e-5, theta);
			}
			System.out.println("Paramters for batchjob optimized, iteration: "+i+" completed");	

			//Storing paramters to start from this iteration again and test current accuray
			if (i%5==0) {
				try {
					//Test
					// Load test data to calculate predictions
					INDArray best_theresholds = t.computeBestThresholds(new DenseDoubleMatrix1D(theta), df.getDevTripples());
					System.out.println("Best theresholds: "+best_theresholds);
					
					//Calculate accuracy of the predictions and accuracy
					INDArray predictions = t.getPrediction(new DenseDoubleMatrix1D(theta), df.getTestTripples(), best_theresholds);
					if(i%25==0){
						Nd4j.writeTxt( u.convertDoubleArrayToFlattenedINDArray(theta), theta_save_path+"//theta_opt_iteration_"+i+".txt", ",");
					}
						
				} catch (Exception e) {
					System.out.println("Error: Saving of weights is not possible!");
				}
			}	
		}
			
		// save optimized theta paramters
		Nd4j.writeTxt(u.convertDoubleArrayToFlattenedINDArray(theta) , theta_save_path+"//theta_opt"+Calendar.getInstance().get(Calendar.DATE)+".txt", ",");
		System.out.println("Training finished, model saved!");
		
		//Test
		System.out.println("Accuracy Test starting with "+theta_load_path);
		// Load test data to calculate predictions
		INDArray best_theresholds = t.computeBestThresholds(new DenseDoubleMatrix1D(theta), df.getDevTripples());
		System.out.println("Best theresholds: "+best_theresholds);
		
		//Calculate accuracy of the predictions and accuracy
		INDArray predictions = t.getPrediction(new DenseDoubleMatrix1D(theta), df.getTestTripples(), best_theresholds);
		
	}
}
