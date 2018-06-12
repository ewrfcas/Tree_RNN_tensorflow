import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.*;

import java.io.*;
import java.util.Properties;

public class Sentence2Tree_step1 {

    // 文本预处理
    private String preprocess(String sen, int type){
        if(type==0){
            sen=sen.replaceAll("--OOV--","UNKNOWNQX");
        }else{
            sen=sen.replaceAll("UNKNOWNQX","--OOV--");
        }
        return sen;
    }

    public static void main(String[] args) {
        // set up pipeline properties
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,parse");
        props.setProperty("parse.maxlen", "400");
        props.setProperty("parse.binaryTrees","true");
        // set up Stanford CoreNLP pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Main main=new Main();
        File file1 = new File("../dataset/dev_word_sen.txt");
        File file2 = new File("../dataset/dev_sen0.txt");
        int len_limit = 100;
        try{
            if(!file1.exists()){
                System.out.print("error");
            }
            if(!file2.exists()){
                file2.createNewFile();
            }
            FileInputStream fileInputStream = new FileInputStream(file1);
            InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            FileWriter fw = new FileWriter(file2,false);
            BufferedWriter bw = new BufferedWriter(fw);
            String text = null;
            int index=0;
            int text_len=0;
            Tree tree;
            String treeString="";
            Annotation annotation = new Annotation();
            while((text = bufferedReader.readLine()) != null){
                // build annotation for a review
                text_len = text.split(" ").length;
                if(text_len>len_limit) {
                    System.out.println(index +"("+ String.valueOf(text_len) + ")"+ " too long failed");
                }
                else{
                    text=main.preprocess(text,0);
                    try {
                        annotation.set(CoreAnnotations.TextAnnotation.class, text);
                        // annotate
                        pipeline.annotate(annotation);
                        //get tree
                        tree = annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0).get(TreeCoreAnnotations.BinarizedTreeAnnotation.class);
                        treeString = tree.toString();
                        treeString = main.preprocess(treeString,1);
                        bw.write(treeString + " " + String.valueOf(index));
                        bw.write("\n");
                        System.out.println(index+"("+ String.valueOf(text_len) + ")" + " success");
                    }catch (Exception e){
                        System.out.println(index+" failed");
                    }
                }
                index++;
            }
            bw.close();
            fw.close();
        }catch (Exception e) {
            //Todo exception
            System.out.print("error");
        }
    }
}