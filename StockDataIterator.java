package org.deeplearning4j.examples.recurrent.indexlstm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class StockDataIterator implements DataSetIterator {

	private static final int VECTOR_SIZE = 10;

	// 每批次的训练数据组数
	private int batchNum;

	// 每组训练数据长度(DailyData的个数)
	private int exampleLength;

	// 数据集
	private List<DailyData> dataList;

	// 存放剩余数据组的index信息
	private List<Integer> dataRecord;

	private double[] maxNum;

	private int outDim;

	public StockDataIterator() {
		this.dataRecord = new ArrayList<>();
	}

	/**
	 * 加载数据并初始化
	 */
	public boolean loadData(String fileName, int batchNum, int exampleLength, int outDim) {
	    this.outDim = outDim;
		this.batchNum = batchNum;
		this.exampleLength = exampleLength;
		maxNum = new double[VECTOR_SIZE];
		// 加载文件中的股票数据
		try {
			this.dataList = readDataFromFile(fileName);
//			System.out.println("finished loading data, datalist size:"+dataList.size());
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		// 重置训练批次列表
		resetDataRecord();
		return true;
	}

	/**
	 * 从文件中读取股票数据
	 */
	public List<DailyData> readDataFromFile(String fileName) throws IOException {
		List<DailyData> dataList = new ArrayList<>();
		FileInputStream fis = new FileInputStream(fileName);
		BufferedReader in = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
		in.readLine();//skip the first line
//		String line = in.readLine();
		for (int i = 0; i < maxNum.length; i++) {
			maxNum[i] = 0;
		}
		System.out.println("reading data..");

		String line = null;
		while ((line=in.readLine()) != null) {
			String[] strArr = line.split("\\s+");
//			System.out.println(strArr.length);
			if (strArr.length == 10) {
				DailyData data = new DailyData();
				// 获得最大值信息，用于归一化
				double[] nums = new double[VECTOR_SIZE];
				for (int j = 0; j < nums.length; j++) {
					nums[j] = Double.valueOf(strArr[j]);
					if (nums[j] > maxNum[j]) {
						maxNum[j] = nums[j];
					}
				}
				// 构造data对象
				data.setPrice(Double.valueOf(nums[0]));
				data.setTIPS_5y(Double.valueOf(nums[1]));
				data.settips_10y(Double.valueOf(nums[2]));
				data.settips_20y(Double.valueOf(nums[3]));
				data.settips_30y(Double.valueOf(nums[4]));
				data.settips_long(Double.valueOf(nums[5]));
                data.setust_bill_10y(Double.valueOf(nums[6]));
                data.setlibor_overnight(Double.valueOf(nums[7]));
                data.setspdr(Double.valueOf(nums[8]));
                data.setusd_cny(Double.valueOf(nums[9]));
				dataList.add(data);

			}
		}
		in.close();
		fis.close();
//		System.out.println("反转list...");
//		Collections.reverse(dataList);
		return dataList;
	}

	/**
	 * 重置训练批次列表
	 */
	private void resetDataRecord() {
		dataRecord.clear();
		int total = dataList.size() / exampleLength + 1;
		for (int i = 0; i < total; i++) {
			dataRecord.add(i * exampleLength);
		}
	}

	public double[] getMaxArr() {
		return this.maxNum;
	}

	@Override
	public boolean hasNext() {
		return dataRecord.size() > 0;
	}

	@Override
	public DataSet next() {
		return next(batchNum);
	}

	@Override
	public DataSet next(int num) {
		if (dataRecord.size() <= 0) {
			throw new NoSuchElementException();
		}
		int actualBatchSize = Math.min(num, dataRecord.size());
//		System.out.println("example len:"+exampleLength+", dataList size:"+dataList.size()+", dataRecord:"+dataRecord.get(0));
		int actualLength = Math.min(exampleLength, dataList.size() - dataRecord.get(0) - 1);
		INDArray input = Nd4j.create(new int[] { actualBatchSize, VECTOR_SIZE, actualLength }, 'f');
		INDArray label = Nd4j.create(new int[] { actualBatchSize, outDim, actualLength }, 'f');
		DailyData nextData = null, curData = null;
		// 获取每批次的训练数据和标签数据
		for (int i = 0; i < actualBatchSize; i++) {
			int index = dataRecord.remove(0);
			int endIndex = Math.min(index + exampleLength, dataList.size() - 1);
			curData = dataList.get(index);
			for (int j = index; j < endIndex; j++) {
				// 获取数据信息
				nextData = dataList.get(j + 1);
				// 构造训练向量
				int c = endIndex - j - 1;
				input.putScalar(new int[] { i, 0, c }, curData.getPrice() / maxNum[0]);
				input.putScalar(new int[] { i, 1, c }, curData.getTIPS_5y() / maxNum[1]);
				input.putScalar(new int[] { i, 2, c }, curData.gettips_10y() / maxNum[2]);
				input.putScalar(new int[] { i, 3, c }, curData.gettips_20y() / maxNum[3]);
				input.putScalar(new int[] { i, 4, c }, curData.gettips_30y() / maxNum[4]);
				input.putScalar(new int[] { i, 5, c }, curData.gettips_long() / maxNum[5]);
                input.putScalar(new int[] { i, 6, c }, curData.getust_bill_10y() / maxNum[6]);
                input.putScalar(new int[] { i, 7, c }, curData.getlibor_overnight() / maxNum[7]);
                input.putScalar(new int[] { i, 8, c }, curData.getspdr() / maxNum[8]);
                input.putScalar(new int[] { i, 9, c }, curData.getusd_cny() / maxNum[9]);
				// 构造label向量
                if(outDim==1){
                    label.putScalar(new int[] { i, 0, c }, nextData.getPrice() / maxNum[0]);
                }else if (outDim==2){
                    if(curData.getPrice()>=nextData.getPrice()){
                        label.putScalar(new int[] {i,0,c}, 1);
                        label.putScalar(new int[] {i,1,c}, 0);
                    }else{
                        label.putScalar(new int[] {i,0,c}, 0);
                        label.putScalar(new int[] {i,1,c}, 1);
                    }

                }


				curData = nextData;
			}
			if (dataRecord.size() <= 0) {
				break;
			}
		}
		return new DataSet(input, label);
	}

	@Override
	public int totalExamples() {
		return (dataList.size()) / exampleLength;
	}

	@Override
	public int inputColumns() {
		return dataList.size();
	}

	@Override
	public int totalOutcomes() {
		return 1;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public void reset() {
		resetDataRecord();
	}

	@Override
	public int batch() {
		return batchNum;
	}

	@Override
	public int cursor() {
		return totalExamples() - dataRecord.size();
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

}
