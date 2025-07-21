FILE=$1

if [ $FILE == "pretrained-network" ]; then
    URL=https://www.dropbox.com/scl/fi/b8eburiyj7k46l2g5efht/model_ours.zip?rlkey=txqqj4732c2krb6690na1m9l6&dl=0
    ZIP_FILE=./model_ours.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE
    rm $ZIP_FILE
    URL=https://www.dropbox.com/scl/fi/j0um6jigf9g24idlq5692/Neutral_AverageJoe2Neutral_Princess.zip?rlkey=d7chnwdnpz41skzwo1hvdt5dv&dl=0
    ZIP_FILE=./Neutral_AverageJoe2Neutral_Princess.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE
    rm $ZIP_FILE

elif  [ $FILE == "datasets" ]; then
    URL=https://www.dropbox.com/scl/fi/debqr38g2yzh92wvw75lz/bvh.zip?rlkey=lau6sc9yzab0ojdfx1elccizc&dl=0
    ZIP_FILE=./bvh.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE
    rm $ZIP_FILE
    URL=https://www.dropbox.com/scl/fi/y7j3iu5cioe2eq9tqa726/datasets.zip?rlkey=5ixrysymz1ougb1xvw11c533w&dl=0
    ZIP_FILE=./datasets.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-network, and bvh."
    exit 1

fi