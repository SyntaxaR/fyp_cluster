# Check $POETRY_HOME and set PATH: $HOME/.local/bin if no
if [ -z "$POETRY_HOME" ]; then
    export POETRY_HOME="$HOME/.local/share/pypoetry"
    export PATH="$POETRY_HOME/bin:$PATH"
else
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Poetry installation
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Install? (y/n)"
    read -r install_poetry
    if [ "$install_poetry" = "y" ]; then
        curl -sSL https://install.python-poetry.org | python3 -
        if ! command -v poetry &> /dev/null; then
            echo "Poetry installation failed."
            exit 1
        else
            echo "Poetry installed successfully."
        fi
    else
        echo "Poetry installation skipped."
        exit 1
    fi
fi

# Install dependencies
poetry install

# Run worker
poetry run python ./src/worker/worker.py