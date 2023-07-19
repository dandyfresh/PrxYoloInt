/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.src.yolo.detec;

import com.src.opencv.core.Constantes;
import com.src.opencv.detec.TrainerSRC;
import com.src.opencv.dto.Annotation;
import com.src.yolo.models.YoloModel;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.core.*;
import static org.opencv.core.Core.FILLED;
import org.opencv.core.Core.MinMaxLocResult;
import static org.opencv.dnn.Dnn.DNN_BACKEND_OPENCV;
import static org.opencv.dnn.Dnn.DNN_TARGET_CPU;
import org.opencv.imgcodecs.Imgcodecs;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_UNCHANGED;// CV_LOAD_IMAGE_UNCHANGED;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import org.opencv.imgproc.Imgproc;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

/**
 *
 * @author desarrollo
 */
public class YoloPrueba {

    public static List<String> classNames;
    public static float confThreshold = 0.5f; // Confidence threshold
    public static float nmsThreshold = 0.4f;  // Non-maximum suppression threshold

    private static final String PATH_TO_SAVE_RESULTS = "D:\\Praxis\\Imagenes\\YOLO\\Images\\Results";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        Test();
        //TestFlow();
//        String pathOpenCV_4_0 = "C:\\temp\\open\\install\\java\\opencv_java420.dll";
//        SRCYolo mod = new SRCYolo();
//        mod.Init(pathOpenCV_4_0);
//        YoloModel model = new YoloModel();
//
//        // ---------- MODELO ECUADOR ----------
//        model.setWeightModel("C:\\datos\\mktbdint\\data\\yolo\\COCO\\yolov3.weights");
//        model.setConfigModelPath("C:\\datos\\mktbdint\\data\\yolo\\COCO\\yolov3.cfg");
//        model.setItems(model.getItemsFromMapNames("C:\\datos\\mktbdint\\data\\yolo\\COCO\\yolov3-spp.names"));
//        File fAnn = new File("C:\\temp\\pruebas_col\\");
//
//        File archs[] = fAnn.listFiles();
//
//        // ---------- MODELO VIDEOS ----------
//        /*model.setWeightModelPath( "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\videos-yolov3_3000.weights" );
//        model.setConfigModelPath( "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\videos-yolov3.cfg" );
//        model.setItems( SRCYolo.getItemsFromMapNames("D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\videos20y_label_map.names") );
//        
//        List<String> images = Arrays.asList(
//                "D:\\Praxis\\Imagenes\\YOLO\\Images\\Results\\Darknet\\videos20y\\data\\Dataset\\agua_brisa_600_ml_2019-01-15_16-11-17_AguaBrisa600ml-PL_77.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\Images\\Results\\Darknet\\videos20y\\data\\Dataset\\agua_brisa_600_ml_2019-01-15_16-11-17_AguaBrisa600ml-PL_101.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\Images\\Results\\Darknet\\videos20y\\data\\Dataset\\agua_brisa_600_ml_2019-01-15_16-11-17_AguaBrisa600ml-PL_121.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\Images\\Results\\Darknet\\videos20y\\data\\Dataset\\agua_brisa_600_ml_2019-01-15_16-11-17_AguaBrisa600ml-PL_154.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\zenu_ensalada_lata_300gr_2019-01-15_17-14-33_ZenuArvejasConZanahoria300g_99.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\avena_gloria_canela_200gr_2019-01-15_16-17-49_AvenaGloria200gTetraPack-Canela_54.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\cafe_sello_rojo_tradicional_250gr_2019-01-15_16-20-56_CafeSelloRojo250gTradicional_1.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\chocolate_corona_tradicional_500g_2019-01-15_17-37-15_ChocolateCoronaPastilla500gZippack_6.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\chocolisto_200gr_2019-01-15_16-24-41_ChocoListo200gDoyPack_4.JPEG",
//                "D:\\Praxis\\Imagenes\\YOLO\\VideoPrueba\\chocolyne_splenda_120gr_2019-01-15_16-27-06_ChocoLyneSplenda120g_14.JPEG"
//        );*/
//        for (File f : archs) {
//            
//            String piimg=f.toString();
//            List<Annotation> an = mod.detectObjects(model, Arrays.asList(piimg), new HashMap(),Constantes.TIPO_GEN);
//            Mat img = imread(piimg, IMREAD_UNCHANGED);
//            TrainerSRC.pintarRectangulos(img, an.get(0), 1, null);
//            
//            imwrite(f.getParent() + "/RES_COCO_" + f.getName(), img);
//        }
    }

    public static void Test() throws IOException {
        System.load("D:/java/librerias/opencv/opencv4_5_0_ALL/java/opencv_java450.dll");
        System.setProperty("jna.library.path", "32".equals(System.getProperty("sun.arch.data.model")) ? "lib/win32-x86" : "lib/win32-x86-64");

        System.out.println(" - START -");

       
        String image = "C:\\temp\\pruebas_col\\RES_382.jpeg";
        String weights = "D:\\temp\\modelosnuevos\\T_LAR_VID_CAR_03_02_2021\\frozen_inference_graph.pb";
        String config = "D:\\temp\\modelosnuevos\\T_LAR_VID_CAR_03_02_2021\\T_LAR_VID_CAR_03_02_2021_label_map.pbtxt";


        /*String classYoloNames = "D:\\Praxis\\Imagenes\\YOLO\\darknet-master\\darknet-master\\data\\coco.names";
        String image = "D:\\\\Praxis\\\\Imagenes\\\\YOLO\\\\darknet-master\\\\darknet-master\\\\data\\\\person.jpg";
        String weights = "D:\\Praxis\\Imagenes\\YOLO\\YOLO-Weights\\yolov3.weights";
        String config = "D:\\Praxis\\Imagenes\\YOLO\\darknet-master\\darknet-master\\cfg\\yolov3.cfg";*/
        

        final int IN_WIDTH = 300;
        final int IN_HEIGHT = 300;
        //final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
        //final double THRESHOLD = 0.2;

        Net net = Dnn.readNetFromTensorflow(weights, config);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        Mat frame = imread(image, IMREAD_UNCHANGED);//CV_LOAD_IMAGE_UNCHANGED);

        Size size = new Size(IN_WIDTH, IN_HEIGHT);
        Scalar scalar = new Scalar(0, 0, 0);

        Mat blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);//Dnn.blobFromImage(matImg, 1.0, size, scalar, false, false);

        net.setInput(blob);

        List<Mat> outs = new ArrayList<>();
       // List<String> names = getOutputsNames(net);
        net.forward(outs);

        postprocess(frame, outs);
        saveImage(frame);
        System.out.println(" - END -");
    }

    public static void TestFlow() throws IOException {
        System.load("D:\\java\\librerias\\opencv\\opencv_4_0_1\\build\\java\\x64\\opencv_java401.dll");
        System.setProperty("jna.library.path", "32".equals(System.getProperty("sun.arch.data.model")) ? "lib/win32-x86" : "lib/win32-x86-64");

        System.out.println(" - START -");

        String image = "D:\\Praxis\\Imagenes\\YOLO\\EcuadorPrueba\\YOGURES_ECUADOR_60933.JPEG";
        String pb = "D:\\downloads\\faster_rcnn_resnet50_coco_2018_01_28\\frozen_inference_graph.pb";
        String config = "D:\\mktbdint\\data\\tensor\\modelos_TF\\mecu\\mecu_label_map.pbtxt";

        /*String classYoloNames = "D:\\Praxis\\Imagenes\\YOLO\\darknet-master\\darknet-master\\data\\coco.names";
        String image = "D:\\\\Praxis\\\\Imagenes\\\\YOLO\\\\darknet-master\\\\darknet-master\\\\data\\\\person.jpg";
        String weights = "D:\\Praxis\\Imagenes\\YOLO\\YOLO-Weights\\yolov3.weights";
        String config = "D:\\Praxis\\Imagenes\\YOLO\\darknet-master\\darknet-master\\cfg\\yolov3.cfg";*/
        final int IN_WIDTH = 416;
        final int IN_HEIGHT = 416;
        //final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
        //final double THRESHOLD = 0.2;

        Net net = Dnn.readNetFromTensorflow(pb);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        Mat frame = imread(image, IMREAD_UNCHANGED);//CV_LOAD_IMAGE_UNCHANGED);

        Size size = new Size(IN_WIDTH, IN_HEIGHT);
        Scalar scalar = new Scalar(0, 0, 0);

        Mat blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);//Dnn.blobFromImage(matImg, 1.0, size, scalar, false, false);

        net.setInput(blob);

        List<Mat> outs = new ArrayList<>();
        List<String> names = getOutputsNames(net);
        net.forward(outs, names);

        postprocess(frame, outs);
        saveImage(frame);
        System.out.println(" - END -");
    }

    // Remove the bounding boxes with low confidence using non-maxima suppression
    static void postprocess(Mat frame, List<Mat> outs) {

        int frameHeight = frame.height();
        int frameWidth = frame.width();

        int cols = frame.cols();
        int rows = frame.rows();
        //Mat subFrame = frame.submat(0, rows, 0, cols);

        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();

        for (Mat out : outs) {
            for (int j = 0; j < out.rows(); j++) {
                Mat scores = out.row(j).colRange(5, out.cols());
                MinMaxLocResult maxLoc = Core.minMaxLoc(scores);
                Point classIdPoint = maxLoc.maxLoc;
                double confidence = maxLoc.maxVal;
                if (confidence > confThreshold) {

                    int centerX = (int) (out.get(j, 0)[0] * frame.cols());
                    int centerY = (int) (out.get(j, 1)[0] * frame.rows());
                    int width = (int) (out.get(j, 2)[0] * frame.cols());
                    int height = (int) (out.get(j, 3)[0] * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.add((int) classIdPoint.x);
                    confidences.add((float) confidence);
                    boxes.add(new Rect2d(left, top, width, height));
                }
            }
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        MatOfInt MatIndices = new MatOfInt();
        MatOfRect2d matRectBoxes = new MatOfRect2d(boxes.toArray(new Rect2d[boxes.size()]));
        float[] floatConfidences = new float[confidences.size()];
        for (int i = 0; i < confidences.size(); i++) {
            floatConfidences[i] = confidences.get(i);
        }
        MatOfFloat matFloat = new MatOfFloat(floatConfidences);
        Dnn.NMSBoxes(matRectBoxes, matFloat, confThreshold, nmsThreshold, MatIndices);
        if (MatIndices.rows() > 0) {
            List<Integer> indices = MatIndices.toList();
            for (int i = 0; i < indices.size(); i++) {
                int idx = indices.get(i);
                Rect2d box = boxes.get(idx);
                drawPred(classIds.get(idx), confidences.get(idx),(int) box.x,(int) box.y,
                        (int)box.x + (int)box.width, (int)box.y + (int)box.height, frame);
            }
        } else {
            System.out.println("No se encuentra coincidencias");
        }

    }

    public static void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat frame) {
        System.out.println("classId:" + classId + " conf:" + conf + " left:" + left + " top:" + top + " right:" + right + " bottom:" + bottom);

        Imgproc.rectangle(frame, new Point(left, top),
                new Point(right, bottom),
                new Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        String label = String.format("%.2f", conf);
        label = classNames.get(classId) + ":" + label;
        System.out.println(label);

        int[] baseLine = new int[1];
        Size labelSize = Imgproc.getTextSize(label, 1, 0.5, 1, baseLine);

        List<Integer> b = Arrays.asList(new Integer[]{top, (int) labelSize.height});
        Object test = Collections.max(b);
        top = (int) test;

        // Draw background for label.
        Imgproc.rectangle(frame,
                new Point(left, top - Math.round(1.5 * labelSize.height)),
                new Point(left + Math.round(1.5 * labelSize.width), top + baseLine[0]),
                new Scalar(255, 255, 255), FILLED);
        // Write class name and confidence.
        Imgproc.putText(frame, label, new Point(left, top), FONT_HERSHEY_SIMPLEX, 0.35, new Scalar(0, 0, 0), 1);
    }

    public static void saveImage(Mat img) throws IOException {
        File fileOtput = File.createTempFile("Prueba",
                ".jpeg",
                new File(PATH_TO_SAVE_RESULTS));

        Imgcodecs.imwrite(fileOtput.getAbsolutePath(), img);
    }

    public static List<String> getOutputsNames(Net net) {
        List<String> names = new ArrayList<>();
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        MatOfInt outLayers = net.getUnconnectedOutLayers();
        //get the names of all the layers in the network
        List<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        for (int i = 0; i < outLayers.toList().size(); ++i) {
            names.add(layersNames.get(outLayers.toArray()[i] - 1));
        }
        return names;
    }

    public static List<String> getYoloClassNames(String URI) throws IOException {
        List<String> list;
        try (Stream<String> lines = Files.lines(Paths.get(URI))) {
            list = lines.collect(Collectors.toList());
        }
        return list;
    }

}
