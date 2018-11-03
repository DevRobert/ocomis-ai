package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkBuilderTest {
    private NeuralNetworkBuilder neuralNetworkBuilder;

    @Before
    public void before() {
        this.neuralNetworkBuilder = new NeuralNetworkBuilder();
    }

    @Test
    public void networkMatchesInputLayerSize() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(2));
        this.neuralNetworkBuilder.setOutputLayerSize(new LayerSize(3));
        this.neuralNetworkBuilder.setActivationFunction(new LinearActivationFunction(0.5));

        NeuralNetwork network = neuralNetworkBuilder.buildNetwork();

        Assert.assertEquals(new LayerSize(2), network.getInputLayer().getSize());
    }

    @Test
    public void networkMatchesOutputLayerSize() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(2));
        this.neuralNetworkBuilder.setOutputLayerSize(new LayerSize(3));
        this.neuralNetworkBuilder.setActivationFunction(new LinearActivationFunction(0.5));

        NeuralNetwork network = this.neuralNetworkBuilder.buildNetwork();

        Assert.assertEquals(new LayerSize(3), network.getOutputLayer().getSize());
    }

    @Test
    public void networkMatchesHiddenLayerSizes() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(2));
        this.neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));
        this.neuralNetworkBuilder.addHiddenLayer(new LayerSize(4));
        this.neuralNetworkBuilder.addHiddenLayer(new LayerSize(5));
        this.neuralNetworkBuilder.setActivationFunction(new LogisticActivationFunction());

        NeuralNetwork network = this.neuralNetworkBuilder.buildNetwork();

        Assert.assertEquals(new LayerSize(4), network.getHiddenLayer(0).getSize());
        Assert.assertEquals(new LayerSize(5), network.getHiddenLayer(1).getSize());
    }

    @Test
    public void failsIfInputLayerSizeNotSpecified() {
        try {
            this.neuralNetworkBuilder.buildNetwork();
        }
        catch(RuntimeException e) {
            Assert.assertEquals("The input layer size must be specified.", e.getMessage());
            return;
        }

        Assert.fail("RuntimeException expected.");
    }

    @Test
    public void failsIfOutputLayerSizeNotSpecified() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(3));

        try {
            this.neuralNetworkBuilder.buildNetwork();
        }
        catch(RuntimeException e) {
            Assert.assertEquals("The output layer size must be specified.", e.getMessage());
            return;
        }

        Assert.fail("RuntimeException expected.");
    }

    @Test
    public void failsIfActivationFunctionNotSpecified() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(3));
        this.neuralNetworkBuilder.setOutputLayerSize(new LayerSize(2));

        try {
            this.neuralNetworkBuilder.buildNetwork();
        }
        catch(RuntimeException e) {
            Assert.assertEquals("The activation function must be specified.", e.getMessage());
            return;
        }

        Assert.fail("RuntimeException expected.");
    }
}
