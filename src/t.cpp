#include "llama.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <sstream>

// LLM_CHAT_TEMPLATE_GRANITE

static struct llama_model_params mparams;
static struct llama_model* model;

static struct llama_context_params cparams;
static struct llama_context* ctx;

struct llama_sampler_chain_params sampler_chain_params;
struct llama_sampler * sampler;

static llama_batch batch;

static void init() {
    sampler_chain_params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sampler_chain_params);
    const float temperature = 0.4;
    const uint32_t seed = 1234;
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    batch = llama_batch_init(512, 0, 1);
}

static void deinit() {
    llama_sampler_free(sampler);    
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

static bool load_model(const char* model_path) {
    mparams = llama_model_default_params();
    model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path);
        return false;
    }
    printf("Model loaded\n");
    cparams = llama_context_default_params();
    ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        model = nullptr;
        return false;
    }
    printf("Model loaded and context created.\n");
    return true;
}

static std::string token_to_piece(const struct llama_context * ctx, llama_token token, bool special = false) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }
    return piece;
}

static std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;
    buf << "[ ";
    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }
        auto detokenized = token_to_piece(ctx, token);
        detokenized.erase(
            std::remove_if(
                detokenized.begin(),
                detokenized.end(),
                [](const unsigned char c) { return !std::isprint(c); }),
            detokenized.end());
        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }
    buf << " ]";
    return buf.str();
}

static std::string detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, bool remove_special = false, bool unparse_special = true) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), remove_special, unparse_special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), remove_special, unparse_special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }
    text.resize(n_chars);
    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

static bool inference() {
    // Define the prompt
    const char* prompt = "Write a short story about a cat.";
    int32_t len = strlen(prompt);
    bool add_special = true;
    bool parse_special = true;
    int n_tokens = len + 2 * add_special;
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(model, prompt, len, tokens.data(), tokens.size(), 
                              add_special, parse_special);
    if (n_tokens < 0) {
        fprintf(stderr, "Failed to tokenize the prompt.\n");
        return false;
    }
    printf("prompt: \"%s\"\n", prompt);
    printf("tokens: %s\n", string_from(ctx, tokens).c_str());
    printf("detokenize: \"%s\"\n", detokenize(ctx, tokens).c_str());
    if (llama_model_has_encoder(model)) {
        int enc_input_size = tokens.size();
        llama_token * enc_input_buf = tokens.data();
        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
    }
    uint32_t n_ctx = llama_n_ctx(ctx);
    printf("n_ctx: %d\n", n_ctx);
#if 0    
    if (llama_model_evaluate(ctx, tokens, n_tokens, n_ctx) != 0) {
        fprintf(stderr, "Failed to evaluate the prompt.\n");
        return;
    }
    printf("Prompt evaluated: '%s'\n", prompt);
    // Generate additional tokens
    for (int i = 0; i < max_tokens; i++) {
        // Fetch the next token
        llama_token next_token = llama_sample_next_token(ctx, NULL);
        if (next_token == LLAMA_TOKEN_EOS) {
            printf("\nEnd of stream reached.\n");
            break;
        }
        // Convert token to string and print
        const char* token_str = llama_token_to_str(ctx, next_token);
        if (token_str) {
            printf("%s", token_str);
        }
        // Append the token to the context
        if (llama_model_evaluate(ctx, &next_token, 1, n_ctx) != 0) {
            fprintf(stderr, "Failed to evaluate the token.\n");
            break;
        }
    }
 #endif   
    printf("\n");
    return true;
}

int main(int argc, char** argv) {
    assert(argc > 1);
    ggml_backend_init_best();
    if (!load_model(argv[1])) { return 1; }
    printf("Model loaded and context created.\n");
    init();
    inference();
    deinit();
    return 0;
}

