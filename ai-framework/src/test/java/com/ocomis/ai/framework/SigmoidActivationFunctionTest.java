package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class SigmoidActivationFunctionTest {
    private ActivationFunction activationFunction;

    @Before
    public void before() {
        this.activationFunction = new SigmoidActivationFunction();
    }

    @Test
    public void testMinusOne() {
        double input = -1;
        double expectedOutput = 0.2689414214;
        Assert.assertEquals(expectedOutput, this.activationFunction.calculateOutput(input), 0.000001);
    }

    @Test
    public void testZero() {
        double input = 0;
        double expectedOutput = 0.5;
        Assert.assertEquals(expectedOutput, this.activationFunction.calculateOutput(input), 0.000001);
    }


    @Test
    public void testOne() {
        double input = 1;
        double expectedOutput = 0.7310585786;
        Assert.assertEquals(expectedOutput, this.activationFunction.calculateOutput(input), 0.000001);
    }
}
