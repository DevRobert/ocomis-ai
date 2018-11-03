package com.ocomis.ai.framework;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private NeuralNetworkLayer inputLayer;
    private NeuralNetworkLayer outputLayer;
    private List<NeuralNetworkLayer> hiddenLayers;

    public NeuralNetwork(LayerSize inputLayerSize, LayerSize outputLayerSize, List<LayerSize> hiddenLayerSizes, ActivationFunction activationFunction) {
        this.inputLayer = new NeuralNetworkLayer(inputLayerSize, activationFunction);
        this.outputLayer = new NeuralNetworkLayer(outputLayerSize, activationFunction);

        this.hiddenLayers = new ArrayList<>();

        for(int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayerSizes.size(); hiddenLayerIndex++) {
            NeuralNetworkLayer hiddenLayer = new NeuralNetworkLayer(hiddenLayerSizes.get(hiddenLayerIndex), activationFunction);
            this.hiddenLayers.add(hiddenLayer);
        }
    }

    public NeuralNetworkLayer getInputLayer() {
        return this.inputLayer;
    }

    public NeuralNetworkLayer getOutputLayer() {
        return this.outputLayer;
    }

    public double[] calculateOutput(double[] input) {
        for(int hiddenLayerIndex = 0; hiddenLayerIndex < this.hiddenLayers.size(); hiddenLayerIndex++) {
            input = this.hiddenLayers.get(hiddenLayerIndex).calculateOutput(input);
        }

        return this.outputLayer.calculateOutput(input);
    }

    public NeuralNetworkLayer getHiddenLayer(int hiddenLayerIndex) {
        if(hiddenLayerIndex >= this.hiddenLayers.size()) {
            throw new RuntimeException("The network does not contain the specified hidden layer.");
        }

        return this.hiddenLayers.get(hiddenLayerIndex);
    }
}
