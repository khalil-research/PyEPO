#!/usr/bin/env python
# coding: utf-8

# to get the right reqs file: pipreqs ./spo

# global vars
VENVS_DIR="PATH_TO_VENV"
VENV_NAME="spo"
GRB_VER="9.1.2"
LOGDIR="./logs"

# load module
echo "Load module..."
module purge
module load cuda
module load python/3.7
module load gurobi/$GRB_VER
# check if the license is set
gurobi_cl 1> /dev/null && echo Success || echo Fail
echo ""

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  echo "Install requirements..."

  # install gurobipy
  echo "  Install GurobiPy..."
  cp -r $GUROBI_HOME/ .
  cd $GRB_VER
  python setup.py install
  cd ..
  rm -r $GRB_VER

  # pip install
  echo "  Install SPO requirements..."
  pip install --no-index --upgrade pip
  pip install tqdm
  pip install numpy
  pip install pandas
  pip install matplotlib
  pip install Pyomo==6.1.2
  pip install scipy==1.6
  pip install pathos
  pip install scikit_learn
  pip install --no-index torch torchvision torchtext torchaudio
  pip install -U tensorboard

  echo "  Install submitit..."
  pip install submitit

  echo "  Install auto-sklearn..."
  pip install auto-sklearn
  pip install emcee
  pip install scikit-optimize
  pip install pyDOE

# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""

# tensorboard
echo "Set tensorboard..."
mkdir -p $LOGDIR
tensorboard --logdir=$LOGDIR --host 0.0.0.0 &

# run . create_env.sh
# run tensorboard dev upload tensorboard --logdir
