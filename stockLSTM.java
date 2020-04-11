package org.deeplearning4j.examples.recurrent.indexlstm;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * Created by yupeng on 2017/5/3.
 */
public class stockLSTM {
    private static final int IN_NUM = 10;
    private static final int OUT_NUM = 1;
    private static final int Epochs = 200;

    private static final int lstmLayer1Size = 50;
    private static final int lstmLayer2Size = 100;

    private static final String modelSaveLocation = "F:\\fintech\\data\\model\\model.zip";

    public static MultiLayerNetwork getNetModel(int nIn, int nOut) {
        MultiLayerConfiguration conf = null;

        switch (OUT_NUM){
            case 1:
                conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1).learningRate(0.001)
                    .rmsDecay(0.5)
                    .seed(12345)
                    .regularization(true).l2(0.001)
                    .weightInit(WeightInit.XAVIER)
//            .updater(Updater.RMSPROP).list()
                    .updater(Updater.RMSPROP).list()
                    .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size).activation(Activation.TANH).build())
                    .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size).activation(Activation.TANH).build())
                    .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nIn(lstmLayer2Size).nOut(nOut).build())
                    .pretrain(false)
                    .backprop(true)
                    .build();
                break;
            case 2:
                conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1).learningRate(0.001)
                    .rmsDecay(0.5)
                    .seed(12345)
                    .regularization(true).l2(0.001)
                    .weightInit(WeightInit.XAVIER)
//            .updater(Updater.RMSPROP).list()
                    .updater(Updater.RMSPROP).list()
                    .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size).activation(Activation.SIGMOID).build())
                    .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size).activation(Activation.SIGMOID).build())
                    .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(lstmLayer2Size).nOut(nOut).build())
                    .pretrain(false)
                    .backprop(true)
                    .build();
                break;
            default:
                break;
        }

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(500));

        return net;
    }

    public static void train(MultiLayerNetwork net, StockDataIterator iterator) {
        // 迭代训练
        for (int i = 0; i < Epochs; i++) {
            DataSet dataSet = null;

            int count = 0;
            while (iterator.hasNext()) {
                count++;
                dataSet = iterator.next();
                net.fit(dataSet);
            }
            System.out.println("batch number: "+count);

            iterator.reset();
            System.out.println();
            System.out.println("=================> finished training epoch: " + (i+1));

            //评估
//            StockEvaluation se = new StockEvaluation(iterator, net, OUT_NUM);
//            se.stats();
            //保存更新模型
//            try {
//                ModelSerializer.writeModel(net, new File(modelSaveLocation), true);
//            } catch (IOException e) {
//                // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
            net.rnnClearPreviousState();
        }
    }

    public static void main(String[] args) throws IOException{
//        String inputFile = IndexLSTMModelling.class.getClassLoader().getResource("stocks/sh000001.csv").getPath();
        String inputFile = "F:\\fintech\\data\\ref_data.txt";

        String validFile = "F:\\fintech\\data\\ref_data_valid.txt";

        int batchSize = 8;
        int exampleLength = 60;
        // 初始化深度神经网络
        StockDataIterator train_iterator = new StockDataIterator();
        train_iterator.loadData(inputFile, batchSize, exampleLength,OUT_NUM);

        MultiLayerNetwork net = null;

        net = getNetModel(IN_NUM, OUT_NUM);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        net.setListeners(new StatsListener(statsStorage));
        uiServer.attach(statsStorage);

        train(net, train_iterator);
        //评估对self的性能
//        StockEvaluation self_steval = new StockEvaluation(train_iterator,BaseOperation.loadModel(modelSaveLocation),OUT_NUM,"train");
//        self_steval.saveEvalOuts();
//
//        StockDataIterator valid_iterator = new StockDataIterator();
//        valid_iterator.loadData(validFile,batchSize,exampleLength,OUT_NUM);
//
//        StockEvaluation valid_steval = new StockEvaluation(valid_iterator,BaseOperation.loadModel(modelSaveLocation),OUT_NUM,"valid");
//        valid_steval.saveEvalOuts();
    }
}
