package com.ocomis.ai.framework;

public class LogisticActivationFunction implements ActivationFunction {
    @Override
    public double calculateOutput(double sumInput) {
        return 1.0 / (1.0 + Math.exp(-sumInput));
    }
}
