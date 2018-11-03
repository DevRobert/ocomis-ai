package com.ocomis.ai.framework;

public class NeuralNetworkBuilder {
    private LayerSize inputLayerSize;
    private LayerSize outputLayerSize;

    public void setInputLayerSize(LayerSize inputLayerSize) {
        this.inputLayerSize = inputLayerSize;
    }

    public void setOutputLayerSize(LayerSize outputLayerSize) {
        this.outputLayerSize = outputLayerSize;
    }

    public NeuralNetwork buildNetwork() {
        if(this.inputLayerSize == null) {
            throw new RuntimeException("The input layer size must be specified.");
        }

        if(this.outputLayerSize == null) {
            throw new RuntimeException("The output layer size must be specified.");
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(this.inputLayerSize, this.outputLayerSize);

        return neuralNetwork;
    }
}
