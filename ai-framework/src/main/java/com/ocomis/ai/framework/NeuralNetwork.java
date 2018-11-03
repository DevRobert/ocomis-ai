package com.ocomis.ai.framework;

public class NeuralNetwork {
    private NeuralNetworkLayer inputLayer;
    private NeuralNetworkLayer outputLayer;

    public NeuralNetwork(LayerSize inputLayerSize, LayerSize outputLayerSize) {
        this.inputLayer = new NeuralNetworkLayer(inputLayerSize);
        this.outputLayer = new NeuralNetworkLayer(outputLayerSize);
    }

    public NeuralNetworkLayer getInputLayer() {
        return this.inputLayer;
    }

    public NeuralNetworkLayer getOutputLayer() {
        return this.outputLayer;
    }
}
