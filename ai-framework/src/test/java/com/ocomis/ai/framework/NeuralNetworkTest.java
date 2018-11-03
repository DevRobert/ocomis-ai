package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Test;

public class NeuralNetworkTest {
    @Test
    public void calculateOutput_noHiddenLayers() {
        NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
        neuralNetworkBuilder.setInputLayerSize(new LayerSize(3));
        neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));

        ActivationFunction activationFunction = new LinearActivationFunction(0.5);
        neuralNetworkBuilder.setActivationFunction(activationFunction);

        NeuralNetwork neuralNetwork = neuralNetworkBuilder.buildNetwork();

        double[] input = new double[] { 0.1, 0.2, 0.3 };

        double[][] weights = new double[][] {
                { 0.1, 0.2, 0.3 },
                { 0.4, 0.5, 0.6 }
        };

        neuralNetwork.getOutputLayer().setWeights(weights);

        double[] expectedOutput = new double[] {
                0.5 * (0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3),
                0.5 * (0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.3)
        };

        double[] actualOutput = neuralNetwork.calculateOutput(input);

        Assert.assertArrayEquals(expectedOutput, actualOutput, 0.000001);
    }
}
