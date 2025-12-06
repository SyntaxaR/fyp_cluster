# Add project path to PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Check $POETRY_HOME and set PATH: $HOME/.local/share/pypoetry/bin if no
if [ -z "$POETRY_HOME" ]; then
    export POETRY_HOME="$HOME/.local/share/pypoetry"
    export PATH="$POETRY_HOME/bin:$PATH"
else
    export PATH="$POETRY_HOME/bin:$PATH"
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
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" poetry install

# Run controller
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" poetry run python -m worker.worker