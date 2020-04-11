package org.deeplearning4j.examples.recurrent.indexlstm;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class StockEvaluation {

	private Evaluation eval;

	public ArrayList<Double> resultList = new ArrayList<>();
	public ArrayList<Double> labelList = new ArrayList<>();

	private int totalnum = 10;

//	private String saveOutput = "F:\\fintech\\data\\outputs\\result.txt";
//	private String saveLabel = "F:\\fintech\\data\\outputs\\label.txt";

    private String saveOutput;
    private String saveLabel;

	public StockEvaluation(StockDataIterator iterator, MultiLayerNetwork net, int outDim, String flag) {
        String task = "";
        if(outDim==1) task = "regression";
        if(outDim==2) task = "classification";
        saveOutput = "F:\\fintech\\data\\outputs_"+task+"\\result_"+flag+".txt";
        saveLabel = "F:\\fintech\\data\\outputs_"+task+"\\label_"+flag+".txt";
        int maxCount = 0;
        switch(flag){
            case "train":
                maxCount = 3295;
                break;
            case "valid":
                maxCount = 1100;
                break;
            default:
                break;
        }

//	    int totalnum = 10;
            double[] maxNum = iterator.getMaxArr();
//            eval = new Evaluation();
            if(outDim==1){
                DataSet dataSet = null;
                INDArray predicts = Nd4j.zeros(iterator.inputColumns(), 1);
                INDArray Label = Nd4j.zeros(iterator.inputColumns(),1);

                int index = 0;

                while (iterator.hasNext()) {
                    dataSet = iterator.next();
                    INDArray labels = dataSet.getLabels();

                    int actualBatchSize = labels.shape()[0];

                    for(int oneElem=0;oneElem<actualBatchSize;oneElem++){
                        for (int i = 0; i < dataSet.getFeatureMatrix().shape()[2]; i++) {
//                        guesses.putScalar(new int[] {index, 0}, 1);
                            INDArray initArray = Nd4j.zeros(1, totalnum, 1);
                            for (int j = 0; j < totalnum; j++) {
                                initArray.putScalar(new int[] { 0, j, 0 }, dataSet.getFeatureMatrix().getRow(oneElem).getRow(j).getColumn(i).getDouble(0));
                            }
//                        for(int dims:initArray.shape()) System.out.print(dims+" ");
//                        System.out.println("");
                            INDArray output = net.rnnTimeStep(initArray);

//                        for(int dims:output.shape()) System.out.println(dims+" ");
//                        System.out.println("");

                            net.rnnClearPreviousState();

                            if(index<=maxCount){
                                predicts.putScalar(new int[]{index}, output.getRow(0).getRow(0).getColumn(0).getDouble(0)*maxNum[0]);
                                Label.putScalar(new int[]{index}, labels.getRow(oneElem).getRow(0).getColumn(i).getDouble(0)*maxNum[0]);
                            }

                            index++;
                        }
                    }
                }
                resultList = BaseOperation.INDArray2DoubleList(predicts,1);
                labelList = BaseOperation.INDArray2DoubleList(Label,1);
            }
            if(outDim==2){
                DataSet dataSet = null;
                INDArray predicts = Nd4j.zeros(iterator.inputColumns(), 1);
                INDArray Label = Nd4j.zeros(iterator.inputColumns(),1);

                int index = 0;

                while (iterator.hasNext()) {
                    dataSet = iterator.next();
                    INDArray labels = dataSet.getLabels();

                    int actualBatchSize = labels.shape()[0];

                    for(int oneElem=0;oneElem<actualBatchSize;oneElem++){
                        for (int i = 0; i < dataSet.getFeatureMatrix().shape()[2]; i++) {
//                        guesses.putScalar(new int[] {index, 0}, 1);
                            INDArray initArray = Nd4j.zeros(1, totalnum, 1);
                            for (int j = 0; j < totalnum; j++) {
                                initArray.putScalar(new int[] { 0, j, 0 }, dataSet.getFeatureMatrix().getRow(oneElem).getRow(j).getColumn(i).getDouble(0));
                            }
//                        for(int dims:initArray.shape()) System.out.print(dims+" ");
//                        System.out.println("");
                            INDArray output = net.rnnTimeStep(initArray);

//                        for(int dims:output.shape()) System.out.println(dims+" ");
//                        System.out.println("");

                            net.rnnClearPreviousState();

                            if(index<=maxCount){
                                predicts.putScalar(new int[]{index}, output.getRow(0).getRow(1).getColumn(0).getDouble(0));
                                Label.putScalar(new int[]{index}, labels.getRow(oneElem).getRow(1).getColumn(i).getDouble(0));
                            }

                            index++;
                        }
                    }
                }
                resultList = BaseOperation.INDArray2DoubleList(predicts,1);
                labelList = BaseOperation.INDArray2DoubleList(Label,1);
            }

//            eval.eval(realOutcomes, guesses);
	}

	public void saveEvalOuts(){
	    BaseOperation.writeValidOutput(saveOutput,resultList);
	    BaseOperation.writeValidOutput(saveLabel,labelList);
    }

	public void stats() {
		System.out.println(this.eval.stats());
	}

}
