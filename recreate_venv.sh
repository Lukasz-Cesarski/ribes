VENVS_DIR=${HOME}/.venvs
mkdir -p ${VENVS_DIR}

virtualenv -p python3.7 ${VENVS_DIR}/ribes_venv
source ${VENVS_DIR}/ribes_venv/bin/activate

pip install -r requirements.txt
echo "export PYTHONPATH=`pwd`" >> $VIRTUAL_ENV/bin/activate
