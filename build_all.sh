DIR=$(pwd)
HOME=/home/trifinger

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
cd $HOME/kaolin && rm -rf build *egg* && IGNORE_TORCH_VER=1 pip install -e .
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
