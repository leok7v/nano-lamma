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

static void batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

static void batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
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
    uint32_t n_ctx = llama_n_ctx(ctx);
    printf("n_ctx: %d\n", n_ctx);
    batch_clear(batch);
    for (int i = 0; i < n_tokens; ++i) {
        batch_add(batch, tokens[i], i, {}, false);
    }
    batch.logits[batch.n_tokens - 1] = 1; // Enable logits for the last token
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to evaluate the prompt.\n");
        return false;
    }
    printf("Prompt evaluated: '%s'\n", prompt);

    // Generation loop
    int max_tokens = 128; // Adjust as needed
    for (int i = 0; i < max_tokens; ++i) {
        // Sample the next token using the sampler
        llama_token new_token_id = llama_sampler_sample(sampler, ctx, batch.n_tokens - 1);

        // Check for end-of-sequence
        if (llama_token_is_eog(model, new_token_id)) {
            printf("\nEnd of stream reached.\n");
            break;
        }
        // Convert token to string and print
        std::string token_str = token_to_piece(ctx, new_token_id);
        printf("%s", token_str.c_str());
        // Add new token to batch
        batch_clear(batch);
        batch_add(batch, new_token_id, n_tokens + i, {}, true);
        // Decode the new token
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Failed to evaluate token.\n");
            break;
        }
    }

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

