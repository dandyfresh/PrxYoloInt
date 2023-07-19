/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.src.yolo.detec;

import com.src.opencv.core.Constantes;
import static com.src.opencv.core.Constantes.DNN_BACKEND;
import static com.src.opencv.core.Constantes.DNN_TARGET;
import com.src.opencv.dto.Annotation;
import com.src.opencv.dto.Bndbox;
import com.src.opencv.dto.Objeto;
import com.src.opencv.dto.StadsBasic;
import com.src.opencv.md.dto.item;
import com.src.opencv.md.intf.ModModel;
import com.src.opencv.md.intf.ModeloGrafico;

import com.src.yolo.models.YoloModel;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_UNCHANGED;
import static org.opencv.imgcodecs.Imgcodecs.imread;

/**
 *
 * @author desarrollo
 */
public class SRCYolo extends ModeloGrafico {

    public static float confThreshold = 0.15f; // Confidence threshold
    public static float nmsThreshold = 0.1f;  // Non-maximum suppression threshold
    //To Resize Images
    public static final int IN_WIDTH = 416;
    public static final int IN_HEIGHT = 416;

    @Override
    public void Init(String path) {
        File f = new File(path);
        if (!f.exists()) {
            //          System.out.println("NO EXISTE LIBRERIA OPENCV:" + path);
        }
        try {
            System.load(path);
            System.setProperty("jna.library.path", "32".equals(System.getProperty("sun.arch.data.model")) ? "lib/win32-x86" : "lib/win32-x86-64");
        } catch (Exception e) {
            //          System.out.println(e.getMessage());
        }
    }

    @Override
    public List<Annotation> detectObjects(ModModel model, List<String> imgs, HashMap<String, StadsBasic> datos, int tipo) {
        List<Annotation> res = new ArrayList();
        Net net = ((YoloModel) model).getNet();
        //net=null;
        if (net == null) {
            System.out.println("CARGO MODELO YOLO1:" + (String) model.getWeightModel());
            net = Dnn.readNetFromDarknet(model.getConfigModelPath(), (String) model.getWeightModel());
            net.setPreferableBackend(DNN_BACKEND);
            net.setPreferableTarget(DNN_TARGET);
        }
        Mat frame = null;
        Mat blob = null;

        for (String im : imgs) {
            try {
                Annotation an = new Annotation();
                an.getFilename().setItem(im);
                final String filename = im;
                BufferedImage img = ImageIO.read(new File(filename));
                if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
                    throw new IOException(String.format("Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust", img.getType(), filename));
                }

                frame = imread(im, org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR);//CV_LOAD_IMAGE_UNCHANGED);
                Size size = new Size(IN_WIDTH, IN_HEIGHT);
                Scalar scalar = new Scalar(0, 0, 0);
                an.setAlto(frame.height());
                an.setAncho(frame.width());
                blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);
                net.setInput(blob);

                List<Mat> outs = new ArrayList<>();
                List<String> names = getOutputsNames(net);
                net.forward(outs, names);
                Postprocess((YoloModel) model, an, frame, outs, filename, datos, tipo);

                res.add(an);

            } catch (Exception e) {
                System.out.println("ERROR EJECUCION MODELO TIPO YOLO:" + e);
            } finally {
                if (frame != null) {
                    frame.release();

                }
                if (blob != null) {
                    blob.release();

                }
                //    net=null;
//          System.out.println(e);
            }
        }

        return res;
    }

    // Remove the bounding boxes with low confidence using non-maxima suppression
    Annotation Postprocess(YoloModel model, Annotation an, Mat frame, List<Mat> outs, String filename, HashMap<String, StadsBasic> datos, int tipo) {

        return this.Postprocess(model.getItems(), an, frame, outs, filename, datos, tipo, model.getMinScore());
    }

    Annotation Postprocess(List<item> items, Annotation an, Mat frame, List<Mat> outs, String filename, HashMap<String, StadsBasic> datos, int tipo, double min_prob) {
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();

        for (Mat out : outs) {
            for (int j = 0; j < out.rows(); j++) {
                Mat scores = out.row(j).colRange(5, out.cols());
                MinMaxLocResult maxLoc = Core.minMaxLoc(scores);
                Point classIdPoint = maxLoc.maxLoc;
                double confidence = maxLoc.maxVal;
                if (confidence >= min_prob) {

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
            out.release();
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
                String cn = items.get(classIds.get(idx)).getName();
                Rect2d box = boxes.get(idx);
                int ind = an.getIndxObj(cn);
                StadsBasic sb = datos.get(cn);
                boolean add = true;

                if (sb==null ||(sb != null && (sb.getAnchoMin() == 0 || box.width >= sb.getAnchoMin()) && (sb.getAltoMin() == 0 || box.height >= sb.getAltoMin()))) {
                    
                    if (ind < 0) {
                        Objeto ob = new Objeto();
                        ob.setName(cn);
                        if (sb != null) {
                            ob.setLabel(datos.get(ob.getName()).getLabel());
                        } else {
                            ob.setLabel(cn);
                        }
                        ob.setTipo(tipo);
                        an.getObject().add(ob);
                        ind = an.getObject().size() - 1;
                    } else {
                        if (sb != null) {
                            an.getObject().get(ind).setLabel(sb.getLabel());
                        } else {
                            an.getObject().get(ind).setLabel(an.getObject().get(ind).getName());
                        }
                    }
                    Bndbox bn = new Bndbox();
                    bn.setScore(confidences.get(idx));
                    bn.setId(cn);
                    bn.setDesc(an.getObject().get(ind).getLabel());
                    bn.setXmin((int) (box.x < 0 ? 0 : ((box.x > frame.width()) ? frame.width() : box.x)));
                    bn.setYmin((int) (box.y < 0 ? 0 : ((box.y > frame.height()) ? frame.height() : box.y)));
                    bn.setXmax((int) ((bn.getXmin() + box.width) > frame.width() ? frame.width() : (bn.getXmin() + box.width)));
                    bn.setYmax((int) ((bn.getYmin() + box.height) > frame.height() ? frame.height() : (bn.getYmin() + box.height)));
                    bn.setTipo(tipo);
                    if ((bn.getAncho() >= Constantes.MIN_PIX_BNB && bn.getAlto() >= Constantes.MIN_PIX_BNB)) {
                        double prop = 1;

                        if (sb != null&& sb.getAnchoMin() == sb.getAltoMin() && sb.getAnchoMin() != 0) {
                            if (bn.getAncho() > bn.getAlto()) {
                                int dif = bn.getAncho() - bn.getAlto() / 2;
                                bn.setXmin(bn.getXmin()+dif);
                                bn.setXmax(bn.getXmax()-dif);
                            }else if(bn.getAncho() < bn.getAlto()){
                                int dif = (bn.getAlto() - bn.getAncho()) / 2;
                                bn.setYmin(bn.getYmin()+dif);
                                bn.setYmax(bn.getYmax()-dif);
                            }
                        }

                        if (sb != null) {
                            if (bn.getOrient() == Constantes.OR_HOR) {
                                prop = sb.getStdsObj().getAvgArmW(sb.getOrient());
                            } else {
                                prop = sb.getStdsObj().getAvgArmH(sb.getOrient());
                            }
                        }
                        an.getObject().get(ind).addBndbox(bn, 1, prop);
                        System.out.printf("\tFound %s \t (score: %.4f) \t (xmin: " + bn.getXmin() + " \t ymin: " + bn.getYmin() + " \t xmax: " + bn.getXmax() + " \t ymax: " + bn.getYmax() + ")\n", cn, confidences.get(idx));
                    }

                }
            }
        } else {
            //          System.out.println("No se encuentran objetos dentro de la imagen: " + filename);
        }

        return an;
    }

    private List<String> getOutputsNames(Net net) {
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

//    public String resConflictoTS(Bndbox b1, Bndbox b2, Mat imgC, String sk[]) {
//        Rectangle rec = new Rectangle();
//        rec.setBounds(Math.min(b1.getXmin(), b2.getXmin()), Math.min(b1.getYmin(), b2.getYmin()), Math.max(b1.getAncho(), b2.getAncho()), Math.max(b1.getAlto(), b2.getAlto()));
//        Mat img = ProcImg.cortar(imgC, rec);
//        String res = "";
//        double pesos[] = new double[sk.length];
//        double tol = 0.2;
//        double pa = 1;
//
//        int wmax = (int) (img.width());
//        int hmax = (int) (img.height());
//        int wmin = (int) (img.width() * (1 - tol));
//        int hmin = (int) (img.height() * (1 - tol));
//        List<String> rutModT = TrainingDelegate.getInstance().getAllModelsBySku("TSFLOWMODEL", Lists.newArrayList(sk));
//        byte data[] = new byte[(int) (img.total()
//                * img.channels())];
//        img.get(0, 0, data);
//        List<String> rutasUt = new ArrayList();
//        Annotation ann = new Annotation();
//        for (int z = 0; z < rutModT.size(); z++) {
//            String rmos = rutModT.get(z);
//            if (!rutasUt.contains(rmos)) {
//                try {
//                    Annotation antemp = SRCTSFlow.detectObject(Constantes.modelosTSFLOW.get(rmos), data, img.height(), img.width());
//                    ann.addAnnotation(antemp, 1);
//                } catch (Exception e) {
//                }
//            }
//        }
//        double maxProb = 0;
//
//        if (ann.getObject() != null) {
//            for (int i = 0; i < ann.getObject().size(); i++) {
//                Objeto obj = ann.getObject().get(i);
//
//                for (int j = 0; j < obj.getBndbox().size(); j++) {
//                    if (maxProb < obj.getBndbox().get(j).getScore()) {
//                        maxProb = obj.getBndbox().get(j).getScore();
//                        res = obj.getName();
//                    }
//                }
//            }
//        }
//
//        return res;
//    }
    public Annotation DetectObject(String img, int modelIndex) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Annotation detectObject(ModModel model, byte[] data, int alto, int ancho) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public Bndbox detectObjects(ModModel model, Mat frame, String nameImg, HashMap<String, StadsBasic> datos, int tipo) {
        Net net = ((YoloModel) model).getNet();

        if (net == null) {
            System.out.println("CARGO MODELO YOLO2:" + (String) model.getWeightModel());
            net = Dnn.readNetFromDarknet(model.getConfigModelPath(), (String) model.getWeightModel());
            net.setPreferableBackend(DNN_BACKEND);
            net.setPreferableTarget(DNN_TARGET);
        }

        Annotation an = new Annotation();
        Bndbox res = new Bndbox();
        Mat blob = null;
        try {

            Size size = new Size(IN_WIDTH, IN_HEIGHT);
            Scalar scalar = new Scalar(0, 0, 0);
            an.setAlto(frame.height());
            an.setAncho(frame.width());
            blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);
            net.setInput(blob);

            List<Mat> outs = new ArrayList<>();
            List<String> names = getOutputsNames(net);
            net.forward(outs, names);
            Postprocess((YoloModel) model, an, frame, outs, nameImg, datos, tipo);

            for (Objeto obj : an.getObject()) {
                for (Bndbox bdb : obj.getBndbox()) {
                    if (bdb.getScore() > res.getScore()) {
                        res = bdb;
                    }
                }

            }

        } catch (Exception e) {
            System.out.println(e);
        } finally {

            if (blob != null) {
                blob.release();

            }
//          System.out.println(e);
        }

        return res;
    }

    public Annotation detectObjects(ModModel model, String img, int tipo) {
        Net net = ((YoloModel) model).getNet();
        // net=null;
        if (net == null) {
            System.out.println("CARGO MODELO YOLO3:" + (String) model.getWeightModel());
            net = Dnn.readNetFromDarknet(model.getConfigModelPath(), (String) model.getWeightModel());
            net.setPreferableBackend(DNN_BACKEND);
            net.setPreferableTarget(DNN_TARGET);
        }

        Annotation an = new Annotation();
        Mat frame = null;
        Mat blob = null;
        try {
            frame = imread(img, IMREAD_UNCHANGED);
            if (frame != null && !frame.empty()) {
                Size size = new Size(IN_WIDTH, IN_HEIGHT);
                Scalar scalar = new Scalar(0, 0, 0);
                an.setAlto(frame.height());
                an.setAncho(frame.width());
                blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);
                net.setInput(blob);

                List<Mat> outs = new ArrayList<>();
                List<String> names = getOutputsNames(net);
                net.forward(outs, names);
                Postprocess((YoloModel) model, an, frame, outs, img, new HashMap(), tipo);
            } else {
                System.out.println("ERROR: IMAGEN NO ENCONTRADA");
            }

        } catch (Exception e) {
            System.out.println(e);
        } finally {
            if (frame != null) {
                frame.release();

            }
            if (blob != null) {
                blob.release();

            }
            //    net=null;
//          System.out.println(e);
        }

        return an;
    }

    public Bndbox detectObject(Net net, List<item> items, Mat frame, String nameImg, HashMap<String, StadsBasic> datos, int tipo, double minScore) {

        Annotation an = new Annotation();
        Bndbox res = new Bndbox();

        Mat blob = null;
        try {

            Size size = new Size(IN_WIDTH, IN_HEIGHT);
            Scalar scalar = new Scalar(0, 0, 0);
            an.setAlto(frame.height());
            an.setAncho(frame.width());
            blob = Dnn.blobFromImage(frame, (1.0 / 255.0), size, scalar, true, false);
            net.setInput(blob);

            List<Mat> outs = new ArrayList<>();
            List<String> names = getOutputsNames(net);
            net.forward(outs, names);
            Postprocess(items, an, frame, outs, nameImg, datos, tipo, minScore);

            for (Objeto obj : an.getObject()) {
                for (Bndbox bdb : obj.getBndbox()) {
                    if (bdb.getScore() > res.getScore()) {
                        res = bdb;
                    }
                }

            }
            blob.release();

        } catch (Exception e) {
            //          System.out.println(e);
        } finally {
            if (blob != null) {
                blob.release();

            }
        }

        return res;
    }

    @Override
    public Annotation detectObject(String img, int modelIndex) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String detectFeature(ModModel model, Mat face, String opts) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String detectFeature(Net n, List<item> itms, Mat face, String opts) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
