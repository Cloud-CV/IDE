# Install the Python runtime dependencies via miniconda (this is much faster
# than using pip for everything).
export PATH=$CONDA_DIR/bin:$PATH

#rm -rf $CONDA_DIR
# clear any cached conda (see #3786), this is true for any public repository!
if [ ! -d $CONDA_DIR ]; then
    if [ "$PYTHON_VERSION" -eq "3" ]; then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    else
        wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    fi
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $CONDA_DIR
    chmod 775 $CONDA_DIR -R

    export PATH="$CONDA_DIR/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda

    conda install pip

    echo $(python -V)
    echo $(python -c "import sys; print sys.path")
    conda update --yes conda
    # The version of boost we're using for Python 3 depends on 3.4 for now.
    if [ "$PYTHON_VERSION" -eq "3" ]; then
        conda install --yes python=3.4
    else
        conda install --yes python=2.7
    fi
fi
