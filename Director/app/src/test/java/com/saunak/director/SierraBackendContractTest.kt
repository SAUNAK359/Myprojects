package com.saunak.director

import com.saunak.director.backend.SierraCreditPolicy
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraBackendContractTest {
    @Test fun chargesOnlySuccessfulCloudReadyCapture() {
        assertTrue(SierraCreditPolicy.shouldConsumeCredit(finalCaptureSucceeded = true, cloudJobReachedReady = true))
        assertFalse(SierraCreditPolicy.shouldConsumeCredit(finalCaptureSucceeded = false, cloudJobReachedReady = true))
        assertFalse(SierraCreditPolicy.shouldConsumeCredit(finalCaptureSucceeded = true, cloudJobReachedReady = false))
        assertFalse(SierraCreditPolicy.shouldConsumeCredit(finalCaptureSucceeded = false, cloudJobReachedReady = false))
    }

    @Test fun oneCreditEqualsOnePicture() {
        assertTrue(SierraCreditPolicy.CREDITS_PER_SUCCESSFUL_AI_PICTURE == 1)
    }
}
