#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {
constexpr const char * TAG = "DirectorVlm";

struct Runtime {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    mtmd_context * vision = nullptr;
    int batch = 512;
    ~Runtime() {
        if (vision) mtmd_free(vision);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

struct SamplerDeleter { void operator()(llama_sampler * value) const { if (value) llama_sampler_free(value); } };
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

struct BatchGuard {
    llama_batch value;
    explicit BatchGuard(int32_t tokens) : value(llama_batch_init(tokens, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(value); }
    BatchGuard(const BatchGuard &) = delete;
    BatchGuard & operator=(const BatchGuard &) = delete;
};

std::unique_ptr<Runtime> runtime;
std::mutex runtime_mutex;
std::atomic_bool cancelled{false};
std::once_flag backend_once;

void check_cancelled() {
    if (cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("generation_cancelled");
}

void fail(JNIEnv * env, const std::string & message) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", message.c_str());
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls) {
        env->ThrowNew(cls, message.c_str());
        env->DeleteLocalRef(cls);
    }
}

std::string jstr(JNIEnv * env, jstring value) {
    if (!value) throw std::runtime_error("Null string passed to native runtime");
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) throw std::runtime_error("Unable to read Java string");
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

jstring make_jstr(JNIEnv * env, const std::string & value) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(value.size()));
    if (!bytes) throw std::runtime_error("Unable to allocate output string buffer");
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(value.size()), reinterpret_cast<const jbyte *>(value.data()));
    jclass string_class = env->FindClass("java/lang/String");
    if (!string_class) { env->DeleteLocalRef(bytes); throw std::runtime_error("Unable to resolve java.lang.String"); }
    jmethodID constructor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    if (!constructor) { env->DeleteLocalRef(bytes); env->DeleteLocalRef(string_class); throw std::runtime_error("Unable to resolve String constructor"); }
    jstring utf8 = env->NewStringUTF("UTF-8");
    jstring out = static_cast<jstring>(env->NewObject(string_class, constructor, bytes, utf8));
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(utf8);
    env->DeleteLocalRef(string_class);
    if (!out) throw std::runtime_error("Unable to create output string");
    return out;
}

std::vector<unsigned char> argb_to_rgb(JNIEnv * env, jintArray pixels, int width, int height) {
    if (!pixels || width <= 0 || height <= 0 || env->GetArrayLength(pixels) != width * height) throw std::runtime_error("Invalid image buffer");
    jint * src = env->GetIntArrayElements(pixels, nullptr);
    if (!src) throw std::runtime_error("Unable to access image pixels");
    std::vector<unsigned char> rgb(static_cast<size_t>(width) * height * 3);
    for (int i = 0; i < width * height; ++i) {
        const uint32_t px = static_cast<uint32_t>(src[i]);
        rgb[3 * i] = static_cast<unsigned char>((px >> 16) & 0xff);
        rgb[3 * i + 1] = static_cast<unsigned char>((px >> 8) & 0xff);
        rgb[3 * i + 2] = static_cast<unsigned char>(px & 0xff);
    }
    env->ReleaseIntArrayElements(pixels, src, JNI_ABORT);
    return rgb;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(256);
    int n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    if (n < 0) throw std::runtime_error("Failed to decode generated token");
    return std::string(buf.data(), static_cast<size_t>(n));
}

std::string format_user_prompt(const llama_model * model, const std::string & content) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl || !*tmpl) throw std::runtime_error("Model chat template unavailable");

    llama_chat_message message{"user", content.c_str()};
    std::vector<char> buffer(content.size() * 2 + 1024);
    int32_t n = llama_chat_apply_template(tmpl, &message, 1, true, buffer.data(), static_cast<int32_t>(buffer.size()));
    if (n < 0) throw std::runtime_error("Failed to apply model chat template");
    if (static_cast<size_t>(n) >= buffer.size()) {
        buffer.resize(static_cast<size_t>(n) + 1);
        n = llama_chat_apply_template(tmpl, &message, 1, true, buffer.data(), static_cast<int32_t>(buffer.size()));
        if (n < 0 || static_cast<size_t>(n) >= buffer.size()) throw std::runtime_error("Failed to resize model chat template buffer");
    }
    return std::string(buffer.data(), static_cast<size_t>(n));
}

void fill_single_token_batch(llama_batch & batch, llama_token token, llama_pos pos) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_saunak_director_ai_NativeVlmBridge_loadModel(JNIEnv * env, jobject, jstring model_path_value, jstring projector_path_value, jint context_size, jint threads) {
    std::lock_guard<std::mutex> guard(runtime_mutex);
    try {
        std::call_once(backend_once, [] { ggml_backend_load_all(); llama_backend_init(); });
        const std::string model_path = jstr(env, model_path_value);
        const std::string projector_path = jstr(env, projector_path_value);
        auto next = std::make_unique<Runtime>();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        next->model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!next->model) throw std::runtime_error("Failed to load Qwen GGUF model");
        if (!llama_model_chat_template(next->model, nullptr)) throw std::runtime_error("Model chat template unavailable");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(context_size);
        cp.n_batch = static_cast<uint32_t>(next->batch);
        cp.n_ubatch = static_cast<uint32_t>(next->batch);
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        next->context = llama_init_from_model(next->model, cp);
        if (!next->context) throw std::runtime_error("Failed to create llama.cpp context");

        mtmd_context_params vp = mtmd_context_params_default();
        vp.use_gpu = false;
        vp.n_threads = threads;
        vp.warmup = false;
        vp.print_timings = false;
        next->vision = mtmd_init_from_file(projector_path.c_str(), next->model, vp);
        if (!next->vision || !mtmd_support_vision(next->vision)) throw std::runtime_error("Failed to load compatible multimodal projector");

        runtime = std::move(next);
        cancelled.store(false, std::memory_order_relaxed);
        return 1;
    } catch (const std::exception & e) {
        runtime.reset();
        fail(env, e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_saunak_director_ai_NativeVlmBridge_generate(JNIEnv * env, jobject, jintArray pixels, jint width, jint height, jstring prompt_value, jstring grammar_value, jint max_tokens, jfloat temperature, jint top_k, jfloat top_p) {
    std::lock_guard<std::mutex> guard(runtime_mutex);
    try {
        if (!runtime) throw std::runtime_error("Director VLM is not loaded");
        cancelled.store(false, std::memory_order_relaxed);
        llama_memory_clear(llama_get_memory(runtime->context), true);
        check_cancelled();

        const std::string prompt = jstr(env, prompt_value);
        const std::string grammar = jstr(env, grammar_value);
        if (grammar.empty()) throw std::runtime_error("Constrained grammar is required");

        auto rgb = argb_to_rgb(env, pixels, width, height);
        check_cancelled();
        mtmd::bitmap image(width, height, rgb.data());
        if (!image.ptr) throw std::runtime_error("Failed to create multimodal image input");

        const std::string marker = mtmd_default_marker();
        const std::string formatted = format_user_prompt(runtime->model, marker + "\n" + prompt);
        const size_t pos = formatted.find(marker);
        if (pos == std::string::npos) throw std::runtime_error("Chat template removed image marker");
        const std::string before = formatted.substr(0, pos);
        const std::string after = formatted.substr(pos + marker.size());

        mtmd_input_text before_text{before.data(), before.size(), false, true};
        mtmd_input_text after_text{after.data(), after.size(), false, true};
        mtmd_input_part p1{&before_text, nullptr};
        mtmd_input_part p2{nullptr, image.ptr.get()};
        mtmd_input_part p3{&after_text, nullptr};
        const mtmd_input_part * parts[]{&p1, &p2, &p3};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        if (!chunks) throw std::runtime_error("Unable to allocate multimodal chunks");
        if (mtmd_tokenize_from_parts(runtime->vision, chunks.get(), parts, 3, true) != 0) throw std::runtime_error("Multimodal tokenization failed");
        check_cancelled();

        llama_pos next_pos = 0;
        if (mtmd_helper_eval_chunks(runtime->vision, runtime->context, chunks.get(), 0, 0, runtime->batch, true, &next_pos) != 0) throw std::runtime_error("Vision/prompt evaluation failed");
        check_cancelled();

        const llama_vocab * vocab = llama_model_get_vocab(runtime->model);
        if (!vocab) throw std::runtime_error("Model vocabulary unavailable");
        SamplerPtr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (!sampler) throw std::runtime_error("Unable to create sampler chain");
        llama_sampler * grammar_sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        if (!grammar_sampler) throw std::runtime_error("Invalid constrained-output grammar");
        llama_sampler_chain_add(sampler.get(), grammar_sampler);
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        BatchGuard batch(1);
        std::string output;
        output.reserve(static_cast<size_t>(max_tokens) * 4);
        int generated = 0;
        for (int i = 0; i < max_tokens; ++i) {
            check_cancelled();
            const llama_token token = llama_sampler_sample(sampler.get(), runtime->context, -1);
            llama_sampler_accept(sampler.get(), token);
            if (llama_vocab_is_eog(vocab, token)) break;
            output += token_piece(vocab, token);
            generated++;
            fill_single_token_batch(batch.value, token, next_pos++);
            check_cancelled();
            if (llama_decode(runtime->context, batch.value) != 0) throw std::runtime_error("Decoder step failed");
        }
        check_cancelled();

        jclass str_cls = env->FindClass("java/lang/String");
        if (!str_cls) throw std::runtime_error("Unable to resolve java.lang.String");
        jobjectArray result = env->NewObjectArray(4, str_cls, nullptr);
        env->DeleteLocalRef(str_cls);
        if (!result) throw std::runtime_error("Unable to allocate generation result");
        jstring text = make_jstr(env, output);
        jstring zero1 = make_jstr(env, "0");
        jstring zero2 = make_jstr(env, "0");
        jstring count = make_jstr(env, std::to_string(generated));
        env->SetObjectArrayElement(result, 0, text);
        env->SetObjectArrayElement(result, 1, zero1);
        env->SetObjectArrayElement(result, 2, zero2);
        env->SetObjectArrayElement(result, 3, count);
        env->DeleteLocalRef(text); env->DeleteLocalRef(zero1); env->DeleteLocalRef(zero2); env->DeleteLocalRef(count);
        return result;
    } catch (const std::exception & e) {
        fail(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_saunak_director_ai_NativeVlmBridge_cancel(JNIEnv *, jobject) {
    cancelled.store(true, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_saunak_director_ai_NativeVlmBridge_close(JNIEnv *, jobject) {
    cancelled.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(runtime_mutex);
    runtime.reset();
}
