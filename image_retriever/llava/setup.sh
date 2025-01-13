cd /content || exit

if [ ! -d "LLaVA" ]; then
    echo "Cloning LLaVA repository..."
    git clone -b v1.0 https://github.com/camenduru/LLaVA
else
    echo "LLaVA repository already exists."
fi

# LLaVA 디렉토리로 이동
cd LLaVA || exit

echo "Installing Python dependencies..."
pip install -r /content/requirements.txt