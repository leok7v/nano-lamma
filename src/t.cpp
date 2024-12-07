#include "llama.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    assert(argc > 1);
    const char* model_path = argv[1];
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path);
        return 1;
    }
    printf("Model loaded\n");
    struct llama_context_params cparams = llama_context_default_params();
    struct llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }
    printf("Model loaded and context created.\n");
    llama_free(ctx);
    llama_free_model(model);
    return 0;
}

