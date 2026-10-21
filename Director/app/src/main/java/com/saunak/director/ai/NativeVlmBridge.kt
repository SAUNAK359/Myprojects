package com.saunak.director.ai

object NativeVlmBridge {
    init { System.loadLibrary("director-vlm") }
    external fun loadModel(modelPath: String, projectorPath: String, contextSize: Int, threads: Int): Long
    external fun generate(pixels: IntArray, width: Int, height: Int, prompt: String, grammar: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float): Array<String>
    external fun cancel()
    external fun close()
}
