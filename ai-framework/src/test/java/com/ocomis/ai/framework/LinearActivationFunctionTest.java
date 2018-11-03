package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Test;

public class LinearActivationFunctionTest {
    @Test
    public void basicTest() {
        ActivationFunction activationFunction = new LinearActivationFunction(0.4);

        Assert.assertEquals(0.9 * 0.4, activationFunction.calculateOutput(0.9), 0.00001);
    }
}
