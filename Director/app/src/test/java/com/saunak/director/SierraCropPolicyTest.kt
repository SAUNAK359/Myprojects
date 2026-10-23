package com.saunak.director

import com.saunak.director.storage.SierraCropPolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraCropPolicyTest {
    @Test fun supportsAllCustomerRatios() {
        listOf("4:3", "3:2", "4:5", "1:1", "16:9", "9:16", "2.39:1").forEach {
            assertTrue("Expected supported ratio $it", SierraCropPolicy.isSupported(it))
        }
    }

    @Test fun rejectsUnknownRatio() {
        assertFalse(SierraCropPolicy.isSupported("7:5"))
    }

    @Test fun cinematicRatioUsesPreciseIntegerRepresentation() {
        assertEquals(239 to 100, SierraCropPolicy.ratioOrNull("2.39:1"))
    }
}
