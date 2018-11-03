package com.ocomis.ai.framework;

import java.util.Objects;

public class LayerSize {
    private int numberOfNeurons;

    public LayerSize(int numberOfNeurons) {
        this.numberOfNeurons = numberOfNeurons;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LayerSize layerSize = (LayerSize) o;
        return numberOfNeurons == layerSize.numberOfNeurons;
    }

    @Override
    public int hashCode() {
        return Objects.hash(numberOfNeurons);
    }
}
