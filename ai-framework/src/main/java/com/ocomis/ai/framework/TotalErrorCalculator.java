package com.ocomis.ai.framework;

public class TotalErrorCalculator {
    public double calculateTotalError(double[] targetOutput, double[] actualOutput) {
        double totalError = 0.0;

        for(int index = 0; index < targetOutput.length; index++) {
            double error = targetOutput[index] - actualOutput[index];
            totalError += error * error;
        }

        return totalError * 0.5;
    }
}
