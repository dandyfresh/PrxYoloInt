/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.src.yolo.models;

import static com.src.opencv.core.Constantes.DNN_BACKEND;
import static com.src.opencv.core.Constantes.DNN_TARGET;
import com.src.opencv.md.dto.item;
import com.src.opencv.md.intf.ModModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

/**
 *
 * @author desarrollo
 */
public class YoloModel implements ModModel {

    private double minScore;
    /**
     * @return the minScore
     */
    public double getMinScore() {
        return minScore;
    }

    /**
     * @param minScore the minScore to set
     */
    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    /**
     * @return the numnets
     */
    public int getNumnets() {
        return numnets;
    }

    /**
     * @param numnets the numnets to set
     */
    public void setNumnets(int numnets) {
        this.numnets = numnets;
    }

    /**
     * @return the net
     */
    public synchronized Net getNet() {

        if (actNet == numnets) {
            actNet = 0;
        }
        return nets.get(actNet++);
    }

    /**
     * @param net the net to set
     */
    public void setNet(Net net) {
        // this.net = net;
    }

    private Object weightModel;
    private String configModelPath;
    private List<item> items;
    
    private int numnets = 1;
    private List<Net> nets = new ArrayList();
    private int actNet = 0;

    @Override
    public Object getWeightModel() {
        return weightModel;
    }

    @Override
    public void setWeightModel(Object weightModelPath) {
        this.weightModel = weightModelPath;
    }

    @Override
    public String getConfigModelPath() {
        return configModelPath;
    }

    @Override
    public void setConfigModelPath(String configModelPath) {
        this.configModelPath = configModelPath;
    }

    @Override
    public List<item> getItems() {
        return items;
    }

    @Override
    public void setItems(List<item> items) {
        this.items = items;
    }

    public void fillNet() {
        nets.clear();
        for (int i = 0; i < this.numnets; i++) {
            nets.add(Dnn.readNetFromDarknet(this.getConfigModelPath(), (String) this.getWeightModel()));
            nets.get(i).setPreferableBackend(DNN_BACKEND);
            nets.get(i).setPreferableTarget(DNN_TARGET);
        }
    }

    @Override
    public String getItemName(int id) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public List<item> getItemsFromMapNames(String pathNames) throws IOException {
        List<item> list = new ArrayList<>();
        List<String> names;
        try (Stream<String> lines = Files.lines(Paths.get(pathNames))) {
            names = lines.collect(Collectors.toList());
            for (int i = 0; i < names.size(); i++) {
                list.add(new item(i, names.get(i)));
            }
        }
        return list;
    }
     @Override
    public String getOpts() {
        return opts;
    }

    @Override
    public void setOpts(String opts) {
        this.opts=opts;
    }
    String opts;
     /**
     * @return the clase
     */
    @Override
    public String getClase() {
        return clase;
    }

    /**
     * @param clase the clase to set
     */
    @Override
    public void setClase(String clase) {
        this.clase = clase;
    }
    private String clase;
    
    
}
