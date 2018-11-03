package com.ocomis.ai.framework;

public class NeuralNetwork {
    private NeuralNetworkLayer inputLayer;
    private NeuralNetworkLayer outputLayer;

    public NeuralNetwork(LayerSize inputLayerSize, LayerSize outputLayerSize, ActivationFunction activationFunction) {
        this.inputLayer = new NeuralNetworkLayer(inputLayerSize, activationFunction);
        this.outputLayer = new NeuralNetworkLayer(outputLayerSize, activationFunction);
    }

    public NeuralNetworkLayer getInputLayer() {
        return this.inputLayer;
    }

    public NeuralNetworkLayer getOutputLayer() {
        return this.outputLayer;
    }

    public double[] calculateOutput(double[] input) {
        return this.outputLayer.calculateOutput(input);
    }
}
