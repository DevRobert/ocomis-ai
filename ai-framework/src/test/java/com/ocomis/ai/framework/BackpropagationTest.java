package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * Test based on: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 */
public class BackpropagationTest {
    private NeuralNetwork neuralNetwork;

    @Before
    public void setUpNetwork() {
        NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
        neuralNetworkBuilder.setInputLayerSize(new LayerSize(2));
        neuralNetworkBuilder.addHiddenLayer(new LayerSize(2));
        neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));
        neuralNetworkBuilder.setActivationFunction(new LogisticActivationFunction());

        this.neuralNetwork = neuralNetworkBuilder.buildNetwork();

        NeuralNetworkLayer hiddenLayer = this.neuralNetwork.getHiddenLayer(0);

        hiddenLayer.setWeights(new double[][] {
                { 0.15, 0.20 },
                { 0.25, 0.30 }
        });

        hiddenLayer.setBias(1.0);
        hiddenLayer.setBiasWeights(new double[] { 0.35, 0.35 });

        NeuralNetworkLayer outputLayer = this.neuralNetwork.getOutputLayer();

        outputLayer.setWeights(new double[][] {
                { 0.40, 0.45 },
                { 0.50, 0.55 }
        });

        outputLayer.setBias(1.0);
        outputLayer.setBiasWeights(new double[] { 0.60, 0.60 });
    }

    @Test
    public void testInitialOutput() {
        double[] input = { 0.05, 0.10 };
        double[] output = this.neuralNetwork.calculateOutput(input);

        Assert.assertEquals(0.751365070, output[0], 0.000000001);
        Assert.assertEquals(0.772928465, output[1], 0.000000001);
    }

    @Test
    public void calculateTotalError() {
        double[] input = { 0.05, 0.10 };
        double[] actualOutput = this.neuralNetwork.calculateOutput(input);
        double[] targetOutput = new double[] { 0.01, 0.99 };

        TotalErrorCalculator totalErrorCalculator = new TotalErrorCalculator();
        double totalError = totalErrorCalculator.calculateTotalError(targetOutput, actualOutput);

        Assert.assertEquals(0.298371109, totalError, 0.000000001);
    }
}
