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

        NeuralNetwork network = neuralNetworkBuilder.buildNetwork();

        Assert.assertEquals(new LayerSize(2), network.getInputLayer().getSize());
    }

    @Test
    public void networkMatchesOutputLayerSize() {
        this.neuralNetworkBuilder.setInputLayerSize(new LayerSize(2));
        this.neuralNetworkBuilder.setOutputLayerSize(new LayerSize(3));

        NeuralNetwork network = this.neuralNetworkBuilder.buildNetwork();

        Assert.assertEquals(new LayerSize(3), network.getOutputLayer().getSize());
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
}
