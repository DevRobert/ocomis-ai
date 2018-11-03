package com.ocomis.ai.framework;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {
    private LayerSize inputLayerSize;
    private LayerSize outputLayerSize;
    private ActivationFunction activationFunction;
    private final List<LayerSize> hiddenLayerSizes = new ArrayList<>();

    public void setInputLayerSize(LayerSize inputLayerSize) {
        this.inputLayerSize = inputLayerSize;
    }

    public void setOutputLayerSize(LayerSize outputLayerSize) {
        this.outputLayerSize = outputLayerSize;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void addHiddenLayer(LayerSize layerSize) {
        this.hiddenLayerSizes.add(layerSize);
    }

    public NeuralNetwork buildNetwork() {
        if(this.inputLayerSize == null) {
            throw new RuntimeException("The input layer size must be specified.");
        }

        if(this.outputLayerSize == null) {
            throw new RuntimeException("The output layer size must be specified.");
        }

        if(this.activationFunction == null) {
            throw new RuntimeException("The activation function must be specified.");
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(this.inputLayerSize, this.outputLayerSize, this.hiddenLayerSizes, this.activationFunction);

        return neuralNetwork;
    }
}
