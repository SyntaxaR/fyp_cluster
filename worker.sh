# Add project path to PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

install_hailo="n"

# Check Hailo accelerator & install dkms & hailo-all if needed
if lspci | grep -i Hailo &> /dev/null; then
    echo "Hailo accelerator detected onboard."
    # Detect dkms & hailo-all installation
    dkms=$(dpkg -s dkms 2> /dev/null | grep "Status: install ok installed" &> /dev/null; echo $?)
    hailo_all=$(dpkg -s hailo-all 2> /dev/null | grep "Status: install ok installed" &> /dev/null; echo $?)
    if [ $dkms -eq 0 ] && [ $hailo_all -eq 0 ]; then
        echo "Hailo dependencies found! Starting worker with Hailo."
    else
        echo "Hailo dependencies not found, install now? (y/n)"
        read -r install_hailo
        if [ "$install_hailo" != "y" ]; then
            echo "Hailo dependencies installation skipped, starting worker without Hailo support."
        else
            echo "Installing Hailo dependencies..."
            sudo apt update
            sudo apt install -y dkms hailo-all
            dkms=$(dpkg -s dkms 2> /dev/null | grep "Status: install ok installed" &> /dev/null; echo $?)
            hailo_all=$(dpkg -s hailo-all 2> /dev/null | grep "Status: install ok installed" &> /dev/null; echo $?)
            if [ $dkms -eq 0 ] && [ $hailo_all -eq 0 ]; then
                echo "Hailo dependencies installed successfully."
            else
                echo "Hailo dependencies installation failed, starting worker without Hailo support."
            fi
        fi
    fi
else
    echo "No Hailo accelerator detected, starting worker without Hailo support."
fi

# Detect PCIe speed /boot/firmware/config.txt, check if there's "dtparam=pciex1_gen=3"
if [ $dkms -eq 0 ] && [ $hailo_all -eq 0 ]; then
    if grep -q "^dtparam=pciex1_gen=3" /boot/firmware/config.txt; then
        echo "Hailo accelerator is set to PCIe Gen3 mode (as recommended by the official documentation)."
        echo "This will improve performance, but might be unstable as RPi5 is NOT certified for Gen3 speed."
    else
        echo "Hailo accelerator is not running in PCIe Gen3 mode."
        echo "Set to Gen3 mode for better performance (might be unstable)? (y/n)"
        read -r set_gen3
        if [ "$set_gen3" = "y" ]; then
            echo "Setting Hailo accelerator to PCIe Gen3 mode..."
            echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
            echo "Please reboot the system for changes to take effect."
            echo "Reboot now? (y/n)"
            read -r reboot_now
            if [ "$reboot_now" = "y" ]; then
                sudo reboot
                exit 0
            else
                echo "Reboot skipped. Please manually reboot to apply changes & restart the program."
                exit 0
            fi
        else
            echo "PCIe Gen3 mode setup skipped, Hailo accelerator will run in PCIe Gen2 mode."
        fi
    fi
fi

if [ "$install_hailo" = "y" ]; then
    echo "Hailo dependencies were installed. Please reboot the system to ensure proper functionality."
    echo "Reboot now? (y/n)"
    read -r reboot_now
    if [ "$reboot_now" = "y" ]; then
        sudo reboot
        exit 0
    else
        echo "Reboot skipped. Please manually reboot to apply changes & restart the program."
        exit 0
    fi
fi

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

# Pause & ask user to connect to cluster network
echo "Connect the worker to the cluster network with a running controller. Press Enter to continue..."
read -r

# Run controller
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" poetry run python -m worker.worker