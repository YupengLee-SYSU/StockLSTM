package org.deeplearning4j.examples.recurrent.indexlstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class BaseOperation {
	public BaseOperation() {
		return;
	}

    public static ArrayList<String> readOneSet2List(String filename){
        ArrayList<String> dataList = new ArrayList<String>();
        try {
            FileReader fr_new = new FileReader(new File(filename));
            BufferedReader br_new = new BufferedReader(fr_new);
            String curline = null;
            while((curline=br_new.readLine())!=null){
                dataList.add(curline);
            }
            br_new.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return dataList;
    }

    public static ArrayList<Double> INDArray2DoubleList(INDArray origNDArray,int index){
        ArrayList<double[]> origDoubleList = NDJ2Double(origNDArray);

        ArrayList<Double> list_new = new ArrayList<>();
        for(int i=0;i<origDoubleList.size();i++){
            list_new.add(origDoubleList.get(i)[index-1]);
        }
        return list_new;
    }

    public static ArrayList<Double> getListOfRowIndex(ArrayList<double[]> origList,int index){
        ArrayList<Double> list_new = new ArrayList<>();
        for(int i=0;i<origList.size();i++){
            list_new.add(origList.get(i)[index-1]);
        }
        return list_new;
    }
    public static boolean isDirEmpty(String Dir){
        File filenew = new File(Dir);
        Boolean flag = false;
        File[] arr = filenew.listFiles();
        if(arr.length==0){
            flag = true;
        }
        return flag;
    }
    public static ArrayList<String> concatData(HashMap<String,ArrayList<String>> dataHash,String[] featSet,ArrayList<String> labelList){
        ArrayList<String> A = dataHash.get(featSet[0]);
        ArrayList<String> B = dataHash.get(featSet[1]);
        ArrayList<String> C = dataHash.get(featSet[2]);
        ArrayList<String> D = dataHash.get(featSet[3]);
        int total_num = A.size();
        ArrayList<String> finalList = new ArrayList<>();
        for(int i=0;i<total_num;i++){
            finalList.add(A.get(i)+","+B.get(i)+","+C.get(i)+","+D.get(i)+","+(int)Double.parseDouble(labelList.get(i)));
        }
        return  finalList;
    }
    public static ArrayList<String> concatData2(HashMap<String,ArrayList<String>> dataHash,String[] featSet,ArrayList<String> labelList){
        ArrayList<String> A = dataHash.get(featSet[0]);
        ArrayList<String> B = dataHash.get(featSet[1]);
        ArrayList<String> C = dataHash.get(featSet[2]);

        int total_num = A.size();
        ArrayList<String> finalList = new ArrayList<>();
        for(int i=0;i<total_num;i++){
            finalList.add(A.get(i)+","+B.get(i)+","+C.get(i)+","+(int)Double.parseDouble(labelList.get(i)));
        }
        return  finalList;
    }

    public static boolean delAllFile(String path) {
        boolean flag = false;
        File file = new File(path);
        if (!file.exists()) {
            return flag;
        }
        if (!file.isDirectory()) {
            return flag;
        }
        String[] tempList = file.list();
        File temp = null;
        for (int i = 0; i < tempList.length; i++) {
            if (path.endsWith(File.separator)) {
                temp = new File(path + tempList[i]);
            } else {
                temp = new File(path + File.separator + tempList[i]);
            }
            if (temp.isFile()) {
                temp.delete();
            }
            if (temp.isDirectory()) {
                delAllFile(path + "/" + tempList[i]);
                flag = true;
            }
        }
        return flag;
    }
	public static void writeEvalLog(String PATH,String logStr) throws IOException{
        FileWriter fw_new = new FileWriter(new File(PATH));
        BufferedWriter bw_new = new BufferedWriter(fw_new);
        bw_new.write(logStr);
        bw_new.close();
    }
	public static ArrayList<String> indarray2feat(INDArray result){
	    ArrayList<String> dataList = new ArrayList<>();
        ArrayList<double[]> resultList = NDJ2Double(result);
        for(int i=0;i<resultList.size();i++){
            double[] tmpResult = resultList.get(i);
            String tmpLine = "";
            for(int j=0;j<tmpResult.length;j++){
                tmpLine = tmpLine+String.valueOf(tmpResult[j])+",";
            }
            dataList.add(tmpLine);
        }
        return dataList;
    }
    public static ArrayList<String> checkAndGetLabel(ArrayList<String> label1,ArrayList<String> label2,ArrayList<String> label3){
        if(!label1.equals(label2) || !label1.equals(label3)){
            System.out.println("wrong: class--BaseOperation--function--checkAndGetLabel");
            System.exit(0);
        }
        return label1;
    }

    public static ArrayList<String> concatThreeList(ArrayList<String> list1,ArrayList<String> list2,ArrayList<String> list3){
	    if(list1.size()!=list2.size() || list1.size()!=list3.size()){
	        System.exit(0);
        }
        ArrayList<String> cList = new ArrayList<>();
        for(int i=0;i<list1.size();i++){
            String tmp_line = list1.get(i)+list2.get(i)+list3.get(i);
            cList.add(tmp_line);
        }
        return cList;
    }
    public static ArrayList<String> concatTwoList(ArrayList<String> list1,ArrayList<String> list2){
        if(list1.size()!=list2.size()) System.exit(0);
        ArrayList<String> cList = new ArrayList<>();
        for(int i=0;i<list1.size();i++){
            String tmp_line = list1.get(i)+(int)Float.parseFloat(list2.get(i));
            cList.add(tmp_line);
        }
        return cList;
    }

    public static ArrayList<String> readDataList(String filename) throws IOException{
	    ArrayList<String> dataList = new ArrayList<>();
        FileReader fr_new = new FileReader(new File(filename));
        BufferedReader br_new = new BufferedReader(fr_new);
        String curline = null;
        while((curline=br_new.readLine())!=null){
            dataList.add(curline);
        }
        return dataList;
    }

    public static ArrayList<String> indarry2label(INDArray labels){
        ArrayList<String> dataList = new ArrayList<>();
        ArrayList<double[]> resultList = NDJ2Double(labels);
        for(int i=0;i<resultList.size();i++){
            double[] tmpResult = resultList.get(i);
            dataList.add(String.valueOf(tmpResult[1]));
        }
        return dataList;
    }


    public static ArrayList<double[]> NDJ2Double(INDArray data){
        ArrayList<double[]> dataList = new ArrayList<>();
        for(int i=0;i<data.rows();i++){
            double[] tmp_array = new double[data.columns()];
            for(int j=0;j<data.columns();j++){
                tmp_array[j] = data.getDouble(i,j);
            }
            dataList.add(tmp_array);
        }
        return  dataList;
    }
    public static void shuffleAndWrite(String origFile,String destFile) throws IOException{
        FileReader fr_new = new FileReader(new File(origFile));
        BufferedReader br_new = new BufferedReader(fr_new);
        String curline = null;
        List<String> dataList = new ArrayList<>();
        while((curline=br_new.readLine())!=null){
            dataList.add(curline);
        }
        br_new.close();
        Collections.shuffle(dataList);

        System.out.println("Orig File:"+origFile+", dest File:"+destFile);
        File p_dest = new File(destFile);
        if(!p_dest.exists()) p_dest.createNewFile();
        FileWriter fw_new = new FileWriter(new File(destFile));
        BufferedWriter bw_new = new BufferedWriter(fw_new);
        for(int i=0;i<dataList.size();i++){
            bw_new.write(dataList.get(i));
            bw_new.newLine();
        }
        bw_new.close();
    }

    public static void deepMLP2Copy(MultiLayerNetwork dbnModel,MultiLayerNetwork mlpModel){
//        if(dbnModel.getnLayers()!=mlpModel.getnLayers()){
//            System.exit(0);
//        }
        for(int i=0;i<mlpModel.getnLayers();i++){
            INDArray tmp_weight = dbnModel.getLayer(i).getParam("W");
            INDArray tmp_bias = dbnModel.getLayer(i).getParam("b");
            mlpModel.getLayer(i).setParam("W",tmp_weight);
            mlpModel.getLayer(i).setParam("b",tmp_bias);
        }
    }

    public static void saveModelParams(String PATH,MultiLayerNetwork myModel) throws IOException{
        FileWriter fw_new = new FileWriter(new File(PATH));
        BufferedWriter bw_new = new BufferedWriter(fw_new);

        File P = new File(PATH);
        if(!P.exists()){
            P.createNewFile();
        }
        int nLayers = myModel.getnLayers();
        for(int i=0;i<nLayers;i++){
            ArrayList<double[]> tmp_weight = NDJ2Double(myModel.getLayer(i).getParam("W"));
            String weight_info = "Type:weight, Layer Number:"+String.valueOf(i)+", Size: ["+String.valueOf(tmp_weight.size())+"x"+String.valueOf(tmp_weight.get(0).length);
            bw_new.write(weight_info);
            bw_new.newLine();
            for(int m=0;m<tmp_weight.size();m++){
                double[] subArray = tmp_weight.get(m);
                for(int n=0;n<subArray.length;n++){
                    bw_new.write(String.valueOf(subArray[n]));
                    bw_new.write("  ");
                }
                bw_new.newLine();
            }
            ArrayList<double[]> tmp_bias = NDJ2Double(myModel.getLayer(i).getParam("b"));
            String bias_info = "Type:bias, Layer Number:"+String.valueOf(i)+", Size: ["+String.valueOf(tmp_bias.size())+"x"+String.valueOf(tmp_bias.get(0).length);
            bw_new.write(bias_info);
            bw_new.newLine();
            for(int m=0;m<tmp_bias.size();m++){
                double[] subArray = tmp_bias.get(m);
                for(int n=0;n<subArray.length;n++){
                    bw_new.write(String.valueOf(subArray[n]));
                    bw_new.write("  ");
                }
                bw_new.newLine();
            }
        }
        bw_new.close();
    }

    public static void saveModel(String PATH, MultiLayerNetwork myModel) throws IOException {
        Boolean saveUpdater = true;
        ModelSerializer.writeModel(myModel,new File(PATH),saveUpdater);
    }
    public static MultiLayerNetwork loadModel(String PATH) throws IOException{
        MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(new File(PATH));
        return restoredModel;
    }

    public static void writeDataList4ND4J(String PATH,ArrayList<double[]> scoreList){
        try {
            File P = new File(PATH);
            if(!P.exists()){
                P.createNewFile();
            }
            BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
            for(int i=0;i<scoreList.size();i++){
                for(int j=0;j<scoreList.get(i).length;j++){
                    BW.write(String.valueOf(scoreList.get(i)[j]));
                    BW.write("  ");
                }
                BW.newLine();
            }
            BW.flush();
            BW.close();
        } catch (IOException e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return;
    }
	public void sortAndGetIndex(float[] origArray,int[] index) {
		for(int q=1;q<origArray.length;q++){
			float temp = origArray[q];
			int temp_i = index[q];
			for(int r=q;r>0&&temp<origArray[r-1];r--){
				origArray[r] = origArray[r-1];
				origArray[r-1] = temp;
				index[r] = index[r-1];
				index[r-1] = temp_i;
			}
		}
	}
	public static String[] readOneSet(String filename){
		ArrayList<String> dataList = new ArrayList<String>();
		try {
			File p = new File(filename);
			if(!p.exists()){
				p.createNewFile();
			}

			FileReader fr_new = new FileReader(p);
			BufferedReader br_new = new BufferedReader(fr_new);
			String curline = null;
			while((curline=br_new.readLine())!=null){
				dataList.add(curline);
			}
			br_new.close();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return dataList.toArray(new String[0]);
	}

	public void sortByIndex(float[] array,int[] newIndex) {
		// TODO Auto-generated method stub
		float[] new_array = new float[array.length];
		for(int e=0;e<array.length;e++){
			new_array[e] = array[newIndex[e]];
		}
		for(int j=0;j<array.length;j++){
			array[j] = new_array[j];
		}
	}
	public void InverseIndex(int[] index){
		int[] new_index = new int[index.length];
		for(int i=0;i<index.length;i++){
			new_index[i] = index[index.length-i-1];
		}
		for(int j=0;j<index.length;j++) index[j] = new_index[j];
		return;
	}
	public static void writeDataList(String PATH,ArrayList<float[]> scoreList){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
			for(int i=0;i<scoreList.size();i++){
				for(int j=0;j<scoreList.get(i).length;j++){
					BW.write(String.valueOf(scoreList.get(i)[j]));
					BW.write("  ");
				}
				BW.newLine();
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return;
	}
	public static void writeDataList4OneHot(String PATH,ArrayList<int[][]> scoreList){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
			for(int i=0;i<scoreList.size();i++){
				int[][] tmp_block = scoreList.get(i);
				for(int j=0;j<tmp_block.length;j++){
					int[] tmp_line = tmp_block[j];
					for(int k=0;k<tmp_line.length;k++){
						BW.write(String.valueOf(tmp_line[k]));
						BW.write("  ");
					}
					BW.newLine();
				}
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return;
	}
	public static void writeDataList4phyChem(String PATH,ArrayList<float[][]> scoreList){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
			for(int i=0;i<scoreList.size();i++){
				float[][] tmp_block = scoreList.get(i);
				for(int j=0;j<tmp_block.length;j++){
					float[] tmp_line = tmp_block[j];
					for(int k=0;k<tmp_line.length;k++){
						BW.write(String.valueOf(tmp_line[k]));
						BW.write("  ");
					}
					BW.newLine();
				}
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return;
	}
	public static void writeDataListDL4J(String PATH,ArrayList<String> scoreList){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
			for(int i=0;i<scoreList.size();i++){
				String tmp_line = scoreList.get(i);
				BW.write(tmp_line);
				BW.newLine();
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return;
	}
	public static void saveData(String PATH,ArrayList<float[][]> scoreList,String dataInfo){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));

			BW.write(dataInfo);
			BW.newLine();
			for(int i=0;i<scoreList.size();i++){
				float[][] tmp_block = scoreList.get(i);
				for(int j=0;j<tmp_block.length;j++){
					float[] tmp_line = tmp_block[j];
					for(int k=0;k<tmp_line.length;k++){
						BW.write(String.valueOf(tmp_line[k]));
						BW.write("  ");
					}
					BW.newLine();
				}
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return;
	}
	public static void writeTXT4Array(String PATH, String[] array){
		try {
			File P = new File(PATH);
			if(!P.exists()){
				P.createNewFile();
			}
			BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
			for(int j=0;j<array.length;j++){
				BW.write(array[j]);
				BW.newLine();
			}
			BW.flush();
			BW.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void makeDir(String tmpDir){
        File F = new File(tmpDir);
        if(!F.exists()) F.mkdirs();
    }
    public static void writeValidOutput(String PATH, ArrayList<Double> dataList){
        try {
            File P = new File(PATH);
            if(!P.exists()){
                P.createNewFile();
            }
            BufferedWriter BW = new BufferedWriter(new FileWriter(new File(PATH)));
            for(int j=0;j<dataList.size();j++){
                BW.write(String.valueOf(dataList.get(j)));
                BW.newLine();
            }
            BW.flush();
            BW.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
