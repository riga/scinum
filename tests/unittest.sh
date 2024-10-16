#!/usr/bin/env bash

# Script to run all tests.
# Arguments:
#   1. The command. Defaults to "python -m pytest tests".

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local repo_dir="$( dirname "${this_dir}" )"

    # default test command
    local cmd="${1:-python -m pytest tests}"

    # execute it
    echo -e "command: \x1b[1;49;39m${cmd}\x1b[0m"
    (
        cd "${repo_dir}"
        eval "${cmd}" && echo -e "\x1b[1;49;32munit tests passed\x1b[0m"
    )
}
action "$@"
