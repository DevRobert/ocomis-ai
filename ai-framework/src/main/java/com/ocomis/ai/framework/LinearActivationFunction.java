package com.ocomis.ai.framework;

public class LinearActivationFunction implements ActivationFunction {
    private final double factor;

    public LinearActivationFunction(double factor) {
        this.factor = factor;
    }

    @Override
    public double calculateOutput(double sumInput) {
        return this.factor * sumInput;
    }
}
