mkdir "input"

kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d input/

kaggle competitions download -c plant-pathology-2021-fgvc8
mkdir "input/plant-pathology-2020-fgvc7
unzip plant-pathology-2020-fgvc7.zip -d input/plant-pathology-2020-fgvc7
mv plant-pathology-2020-fgvc7.zip input/plant-pathology-2020-fgvc7