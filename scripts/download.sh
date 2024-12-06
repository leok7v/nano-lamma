#!/usr/bin/env zsh
if [ ! -f "models/granite-3.0-1b-a400m-instruct/granite-3.0-1B-a400M-instruct-Q8_0.gguf" ]; then
    pushd models/granite-3.0-1b-a400m-instruct || exit 1
    curl -OL https://github.com/leok7v/gyptix/releases/download/model/granite-3.0-1B-a400M-instruct-Q8_0.gguf
    popd
fi

