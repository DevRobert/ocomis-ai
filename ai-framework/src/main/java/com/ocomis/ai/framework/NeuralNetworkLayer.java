package com.ocomis.ai.framework;

public class NeuralNetworkLayer {
    private LayerSize layerSize;
    private double[][] weights;
    private ActivationFunction activationFunction;

    public NeuralNetworkLayer(LayerSize layerSize, ActivationFunction activationFunction) {
        this.layerSize = layerSize;
        this.activationFunction = activationFunction;
    }

    public LayerSize getSize() {
        return this.layerSize;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] calculateOutput(double[] input) {
        double[] output = new double[this.layerSize.getNumberOfNeurons()];

        for(int outputIndex = 0; outputIndex < output.length; outputIndex++) {
            double sumInput = 0.0;

            for(int inputIndex = 0; inputIndex < input.length; inputIndex++) {
                double weight = weights[outputIndex][inputIndex];
                sumInput += input[inputIndex] * weight;
            }

            output[outputIndex] = this.activationFunction.calculateOutput(sumInput);
        }

        return output;
    }
}
