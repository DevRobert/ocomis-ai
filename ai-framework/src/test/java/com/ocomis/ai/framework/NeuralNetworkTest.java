package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Test;

public class NeuralNetworkTest {
    @Test
    public void calculateOutput_noHiddenLayers_noBias() {
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

    @Test
    public void calculateOutput_noHiddenLayers_withBias() {
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

        neuralNetwork.getOutputLayer().setBias(0.8);
        neuralNetwork.getOutputLayer().setBiasWeights(new double[] { 0.5, 0.6 });

        double[] expectedOutput = new double[] {
                0.5 * (0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.8 * 0.5),
                0.5 * (0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.3 + 0.8 * 0.6)
        };

        double[] actualOutput = neuralNetwork.calculateOutput(input);

        Assert.assertArrayEquals(expectedOutput, actualOutput, 0.000001);
    }

    @Test
    public void calculateOutput_oneHiddenLayer_noBias() {
        NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
        neuralNetworkBuilder.setInputLayerSize(new LayerSize(3));
        neuralNetworkBuilder.addHiddenLayer(new LayerSize(2));
        neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));

        ActivationFunction activationFunction = new LinearActivationFunction(0.5);
        neuralNetworkBuilder.setActivationFunction(activationFunction);

        NeuralNetwork neuralNetwork = neuralNetworkBuilder.buildNetwork();

        double[] input = new double[] { 0.1, 0.2, 0.3 };

        double[][] hiddenLayerWeights = new double[][] {
                { 0.1, 0.2, 0.3 },
                { 0.4, 0.5, 0.6 }
        };

        neuralNetwork.getHiddenLayer(0).setWeights(hiddenLayerWeights);

        double[][] outputLayerWeights = new double[][] {
                { 0.15, 0.25 },
                { 0.45, 0.55 }
        };

        neuralNetwork.getOutputLayer().setWeights(outputLayerWeights);

        double[] expectedOutputHiddenLayer = new double[] {
                0.5 * (0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3),
                0.5 * (0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.3)
        };

        double[] expectedOutput = new double[] {
                0.5 * (0.15 * expectedOutputHiddenLayer[0] + 0.25 * expectedOutputHiddenLayer[1]),
                0.5 * (0.45 * expectedOutputHiddenLayer[0] + 0.55 * expectedOutputHiddenLayer[1])
        };

        double[] actualOutput = neuralNetwork.calculateOutput(input);

        Assert.assertArrayEquals(expectedOutput, actualOutput, 0.000001);
    }

    @Test
    public void calculateOutput_twoHiddenLayers_withBias() {
        NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
        neuralNetworkBuilder.setInputLayerSize(new LayerSize(3));
        neuralNetworkBuilder.addHiddenLayer(new LayerSize(2));
        neuralNetworkBuilder.addHiddenLayer(new LayerSize(2));
        neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));

        ActivationFunction activationFunction = new LinearActivationFunction(0.5);
        neuralNetworkBuilder.setActivationFunction(activationFunction);

        NeuralNetwork neuralNetwork = neuralNetworkBuilder.buildNetwork();

        double[] input = new double[] { 0.1, 0.2, 0.3 };

        double[][] firstHiddenLayerWeights = new double[][] {
                { 0.1, 0.2, 0.3 },
                { 0.4, 0.5, 0.6 }
        };

        neuralNetwork.getHiddenLayer(0).setWeights(firstHiddenLayerWeights);

        neuralNetwork.getHiddenLayer(0).setBias(0.6);
        neuralNetwork.getHiddenLayer(0).setBiasWeights(new double[] { 0.1, 0.2 });

        double[][] secondHiddenLayerWeights = new double[][] {
                { 0.12, 0.22 },
                { 0.42, 0.52 }
        };

        neuralNetwork.getHiddenLayer(1).setWeights(secondHiddenLayerWeights);

        neuralNetwork.getHiddenLayer(1).setBias(0.9);
        neuralNetwork.getHiddenLayer(1).setBiasWeights(new double[] { 0.3, 0.4 });

        double[][] outputLayerWeights = new double[][] {
                { 0.15, 0.25 },
                { 0.45, 0.55 }
        };

        neuralNetwork.getOutputLayer().setWeights(outputLayerWeights);

        neuralNetwork.getOutputLayer().setBias(1.0);
        neuralNetwork.getOutputLayer().setBiasWeights(new double[] { 0.5, 0.6 });

        double[] expectedOutputFirstHiddenLayer = new double[] {
                0.5 * (0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.1 * 0.6),
                0.5 * (0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.3 + 0.2 * 0.6)
        };

        double[] expectedOutputSecondHiddenLayer = new double[] {
                0.5 * (0.12 * expectedOutputFirstHiddenLayer[0] + 0.22 * expectedOutputFirstHiddenLayer[1] + 0.3 * 0.9),
                0.5 * (0.42 * expectedOutputFirstHiddenLayer[0] + 0.52 * expectedOutputFirstHiddenLayer[1] + 0.4 * 0.9)
        };

        double[] expectedOutput = new double[] {
                0.5 * (0.15 * expectedOutputSecondHiddenLayer[0] + 0.25 * expectedOutputSecondHiddenLayer[1] + 0.5 * 1.0),
                0.5 * (0.45 * expectedOutputSecondHiddenLayer[0] + 0.55 * expectedOutputSecondHiddenLayer[1] + 0.6 * 1.0)
        };

        double[] actualOutput = neuralNetwork.calculateOutput(input);

        Assert.assertArrayEquals(expectedOutput, actualOutput, 0.000001);
    }

    @Test
    public void getHiddenLayer_fails_ifSpecifiedLayerNotExistent() {
        NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
        neuralNetworkBuilder.setInputLayerSize(new LayerSize(1));
        neuralNetworkBuilder.setOutputLayerSize(new LayerSize(1));
        neuralNetworkBuilder.setActivationFunction(new LogisticActivationFunction());

        NeuralNetwork neuralNetwork = neuralNetworkBuilder.buildNetwork();

        try {
            neuralNetwork.getHiddenLayer(0);
        }
        catch(RuntimeException e) {
            Assert.assertEquals("The network does not contain the specified hidden layer.", e.getMessage());
            return;
        }

        Assert.fail("RuntimeException expected.");
    }
}
