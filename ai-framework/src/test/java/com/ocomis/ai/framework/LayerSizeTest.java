package com.ocomis.ai.framework;

import org.junit.Assert;
import org.junit.Test;

public class LayerSizeTest {
    @Test
    public void equalsReturnsTrueForSameSizes() {
        LayerSize layerSize1 = new LayerSize(2);
        LayerSize layerSize2 = new LayerSize(2);

        Assert.assertTrue(layerSize1.equals(layerSize2));
    }

    @Test
    public void equalsReturnsFalseForDifferenzSizes() {
        LayerSize layerSize1 = new LayerSize(2);
        LayerSize layerSize2 = new LayerSize(3);

        Assert.assertFalse(layerSize1.equals(layerSize2));
    }
}
