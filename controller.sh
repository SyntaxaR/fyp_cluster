# Add project path to PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Check uv installation
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Install? (y/n)"
    read -r install_uv
    if [ "$install_uv" = "y" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        if ! command -v uv &> /dev/null; then
            echo "uv installation failed."
            exit 1
        else
            echo "uv installed successfully."
        fi
    else
        echo "uv installation skipped."
        exit 1
    fi
fi

# Run controller
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" uv run python -m controller.controller