import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;


/**
 * Original from Standford Core NLP, modified to ND4j and my purpose with only a small number of functions.
 * These functions are befored implemented in the Util.java Class but seems better to handle a tensor as additional class.
 * This class defines a block tensor, somewhat like a three
 * dimensional matrix.  This can be created in various ways, such as
 * by providing an array of SimpleMatrix slices, by providing the
 * initial size to create a 0-initialized tensor, or by creating a
 * random matrix.
 *
 * @author John Bauer
 * @author Richard Socher
 */
public class TensorP{
  private final DoubleMatrix2D[] slices;

  final int numRows;
  final int numCols;
  final int numSlices;

  /**
   * Creates a zero initialized tensor
   */
  public TensorP(int numRows, int numCols, int numSlices, int filling) {
    slices = new DoubleMatrix2D[numSlices];
    double r = 1 / Math.sqrt(2*numRows); // r is used for a better initialization of w
    if (filling == 0) {
        for (int i = 0; i < numSlices; ++i) {
            slices[i] =  new DenseDoubleMatrix2D(numRows, numCols).assign(0);
          }
	}else if (filling == 1) {
        for (int i = 0; i < numSlices; ++i) {
        	slices[i] =  new DenseDoubleMatrix2D(numRows, numCols).assign(1);
          }
	}else if(filling== 2){
	    for (int i = 0; i < numSlices; ++i) {
	        slices[i] = DoubleFactory2D.dense.random(numRows,numCols);
	        //System.out.println("Colt: "+slices[i].viewPart(0, 0, 3, 5));
	        slices[i].assign(Nd4j.rand(numRows, numCols).mul(2 * r - r).transpose().data().asDouble());
	        //System.out.println("Colt: "+slices[i].viewPart(0, 0, 3, 5));
	     }
	    
	}

    this.numRows = numRows;
    this.numCols = numCols;
    this.numSlices = numSlices;
  }

  /**
   * Copies the data in the slices.  Slices are copied rather than
   * reusing the original SimpleMatrix objects.  Each slice must be
   * the same size.
   */
  public TensorP(INDArray[] slices) {
	System.out.println("NOT WORKING!");
    this.numRows = slices[0].rows();
    this.numCols = slices[0].columns();
    this.numSlices = slices.length;
    this.slices = new DoubleMatrix2D[slices.length];
    for (int i = 0; i < numSlices; ++i) {
      if (slices[i].rows() != numRows || slices[i].columns() != numCols) {
        throw new IllegalArgumentException("Slice " + i + " has matrix dimensions " + slices[i].rows() + "," + slices[i].columns() + ", expected " + numRows + "," + numCols);
      }
      //this.slices[i] = new SimpleMatrix(slices[i]);
    }
    
  }

  /**
   * Returns a randomly initialized tensor with values draft from the
   * uniform distribution between minValue and maxValue.
   */
  public static TensorP random(int numRows, int numCols, int numSlices) {
    TensorP tensor = new TensorP(numRows, numCols, numSlices, 0);
    for (int i = 0; i < numSlices; ++i) {
      tensor.slices[i] = DoubleFactory2D.dense.random(numRows,numCols);
    }
    return tensor;
  }

  /**
   * Number of rows in the tensor
   */
  public int numRows() {
    return numRows;
  }

  /**
   * Number of columns in the tensor
   */
  public int numCols() {
    return numCols;
  }

  /**
   * Number of slices in the tensor
   */
  public int numSlices() {
    return numSlices;
  }

  /**
   * Total number of elements in the tensor
   */
  public int getNumElements() {
    return numRows * numCols * numSlices;
  }

  
  public void putForAllSlices(int row, int column, double value) {
	    for (int slice = 0; slice < numSlices; ++slice) {
	      slices[slice].set(row, column, value);
	    }
  }
  public void put(int slice, int row, int column, double value) {
	      slices[slice].set(row, column, value);
  }


  /**
   * Use the given <code>matrix</code> in place of <code>slice</code>.
   * Does not copy the <code>matrix</code>, but rather uses the actual object.
   */
  public void setSlice(int slice, DoubleMatrix2D matrix) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    if (matrix.columns() != numCols) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.columns() + " columns, tensor has " + numCols);
    }
    if (matrix.rows() != numRows) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.rows() + " columns, tensor has " + numRows);
    }
    slices[slice] = matrix;
  }

  /**
   * Returns the SimpleMatrix at <code>slice</code>.
   * <br>
   * The actual slice is returned - do not alter this unless you know what you are doing.
   */
  public DoubleMatrix2D getSlice(int slice) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    return slices[slice];
  }
  
  public DoubleMatrix1D as1DArrayLikeNumpy() {
	  //export a Tensor in the same order style as numpy does
	  //System.out.println("size: "+numRows*numCols*numSlices);
	    DoubleMatrix1D flat = new DenseDoubleMatrix1D(numRows*numCols*numSlices);
	    //System.out.println("numRows: "+numRows+"numCols:"+numCols+"numSlices:"+numSlices);
	    int counter=0;
		for (int r = 0; r < (numRows); r++) {
			for (int c = 0; c < (numCols); c++) {
				for (int s = 0; s < numSlices; s++) {
					//System.out.println("s-r-c: "+s+"-"+r+"-"+c+"-");
					flat.set(counter++, slices[s].get(r, c));
				}
			}
		}
		return flat;		
	}
  
  public void fill(DoubleMatrix1D flat) {
	//read in a Tensor from a flat array like numpy

	    int counter=0;
	    if (flat.size() == numRows*numCols*numSlices) {
			for (int r = 0; r < (numRows); r++) {
				for (int c = 0; c < (numCols); c++) {
					for (int s = 0; s < numSlices; s++) {
						slices[s].set(r, c, flat.get(counter++));
					}
				}
			}
		}else{
			System.out.println("ERROR: flat size different!");
			  //System.out.println("size: "+numRows*numCols*numSlices);
		}
	}
  
  public void fillLikeNumpy(DoubleMatrix1D flat) {
	//read in a Tensor from a flat array like numpy

	    int counter=0;
	    if (flat.size() == numRows*numCols*numSlices) {
			for (int r = 0; r < (numRows); r++) {
				for (int c = 0; c < (numCols); c++) {
					for (int s = 0; s < numSlices; s++) {
						slices[s].set(r, c, flat.get(counter++));
					}
				}
			}
		}else{
			System.out.println("ERROR: flat size different!");
			  //System.out.println("size: "+numRows*numCols*numSlices);
		}
	    
	}  

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    for (int slice = 0; slice < numSlices; ++slice) {
      result.append("Slice " + slice + "\n");
      result.append(slices[slice]);
    }
    return result.toString();
  }
}