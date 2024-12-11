#!/bin/bash

wget https://drive.iust.ac.ir/index.php/s/Xd3qqQ69b4k9R23/download/checkpoint.tar.gz.0
wget https://drive.iust.ac.ir/index.php/s/AmeMY3LsrBod8RR/download/checkpoint.tar.gz.1
wget https://drive.iust.ac.ir/index.php/s/Sx4S7WGrwGMPzjX/download/checkpoint.tar.gz.2
wget https://drive.iust.ac.ir/index.php/s/H662GKS2sjbraAM/download/checkpoint.tar.gz.3
wget https://drive.iust.ac.ir/index.php/s/dotBfDJwJfkEyKD/download/checkpoint.tar.gz.4

cat checkpoint.tar.gz.* | tar xzvf -
rm checkpoint.tar.gz.*
