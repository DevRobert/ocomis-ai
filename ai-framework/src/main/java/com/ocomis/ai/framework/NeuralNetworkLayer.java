package com.ocomis.ai.framework;

public class NeuralNetworkLayer {
    private LayerSize layerSize;

    public NeuralNetworkLayer(LayerSize layerSize) {
        this.layerSize = layerSize;
    }

    public LayerSize getSize() {
        return this.layerSize;
    }
}
