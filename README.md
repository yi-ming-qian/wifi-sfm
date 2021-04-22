## Getting Started

Clone the repository:
```bash
git clone https://github.com/yi-ming-qian/wifi-sfm.git
```

We use Python 3.7 and C++ in our implementation, please install dependencies:
```bash
conda create -n wifi python=3.7
conda activate wifi
conda install -c conda-forge ceres-solver
```

## Dataset
Please download our dataset from [here](https://www.dropbox.com/s/0y4mljxln4v1ka1/data.zip?dl=0).

## Single-day correction for RONIN trajectories
Please run the following command:
```bash
cp CMakeLists-single.txt CMakeLists.txt
mkdir build
cd build
cmake ..
make
cd ..
sh run.sh
```

## Multi-day alignment
Please run the following command:
```bash
cp CMakeLists-multi.txt CMakeLists.txt
cd build
rm -r *
cmake ..
make
cd ..
sh ./build/multiAlign
```

The raw output will be saved under the folder "/outputs/". To visualize the results, please run "python main.py" and check "experiments/multialign/joint.png". The results would look like this:
![sample result](https://github.com/yi-ming-qian/wifi-sfm/blob/main/joint%20(copy).png)

## Contact
Please email [https://yi-ming-qian.github.io/](https://yi-ming-qian.github.io/) if you have any problems when running the program. **This porject is still under development.**

## Acknowledgements
We thank Pyojin Kim's repo: [https://github.com/PyojinKim/wifisfm](https://github.com/PyojinKim/wifisfm).