<p align="center">
<img width="400px" src="https://nikolako.net/img/perdif.svg"/>
</p>

# Personalized Diffusions for Recommendation

Personalized Diffusions (PERDIF) is a random-walk-based top-N recommendation approach that combines the advantages of neighborhood- and graph-based collaborative filtering methods. It achieves state-of-the-art recommendation performance and has low computational requirements.

This package provides a C-based multi-threaded implementation of PERDIF that consists of a set of command-line programs. 


## Citing
If you use any part of this software in your research, please cite us using the
following BibTex entry:

```
@inproceedings{PERDIF:2019,
 author = {Nikolakopoulos, Athanasios N. and Berberidis, Dimitris and Karypis, George and Giannakis, Georgios B.},
 title = {Personalized Diffusions for Top-n Recommendation},
 booktitle = {Proceedings of the 13th ACM Conference on Recommender Systems},
 series = {RecSys '19},
 year = {2019},
 isbn = {978-1-4503-6243-6},
 location = {Copenhagen, Denmark},
 pages = {260--268},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3298689.3346985},
 doi = {10.1145/3298689.3346985},
 acmid = {3346985},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {item models, random walks, top-n recommendation},
} 
```


##  Downloading PERDIF

```bash
git clone https://github.com/nikolakopoulos/Personalized-Diffusions.git
```

## Building standalone PERDIF binary 

To build PERDIF you can follow the instructions below:

### Dependencies

General dependencies for building perdif are: gcc, cmake, build-essential, mkl (for blas routines).
For Ubuntu machines on which you have `sudo` privileges, we provided the `depmkl.sh` script that automates the process of obtaining and installing the dependencies, which can be used as follows:

```bash
bash depmkl.sh
source ~/.bashrc 
```

For machines on which you do not have `sudo` privileges, you should download the MKL tarball from [Intel's website](https://software.intel.com/en-us/mkl) and then install it locally using the `install.sh` script they provide. After installing it you should add `your-path-to-intel/intel/mkl/bin/mklvars.sh intel64`in your bashrc and run `source ~/.bashrc`.


### Building and installing PerDIF  

In order to build PerDIF, run:

```bash
make
```

## Getting started

Here are some examples to quickly try out PERDIF on the sample datasets that we provide.


###  Command-line programs
PERDIF can be used by running the following two command-line programs:
- `perdif_learn`: for learning personalized diffusions for each user, and
- `perdif_mselect`: for selecting the base item model as well as the number of random walk steps K.

For example: 
```bash
./perdif_learn -dataset=yahoo -strategy=free -max_walk=3
```
will fit the personalized diffusions using the PerDIF FREE variant, with number of steps = 3. 
The learned diffusion coefficients and the corresponding parameters mu will be stored in the data/out/yahoo folder.

```bash
./perdif_learn -dataset=ml1m -strategy=dictionary -max_walk=6
```
fits a PerDIF PAR model, with number of maximum number of steps = 6. 

For more information regarding available choices  

```bash

 Usage:
   perdif_learn [options]
 
 Options:
 
   -dataset=string
      Specifies the dataset to be used.
        The dataset name is assumed to correspond to the name of the dataset folder in data/in and data/out directories.
        The default value is ml1m
 
   -max_walk=int
      Specifies that length of the personalized item exploration walks.
      The default value is 5
 
   -strategy=string
      Available options are:
        single-best  -  Chooses for each user the Kth step that minimizes training error [default].
        free         -  The PerDIF^Free model.
        dictionary   -  The PerDIF^Par model.
        hk           -  PerDIF^par using only Heat Kernel weights.
        ppr          -  PerDIF^par using only Personalized PageRank weights.
 
   -bpr_fit
      It fits the personalized diffusions using a BRP loss. Default is RMSE
 
   -usr_threads=int
      Specifies the number of threads to be used for learning and evaluating the model.
      The default value is maximum number of threads available on the machine.
 
   -help
      Prints this message.
 
 Example run: ./perdif_learn -dataset=yahoo -max_walk=3 -strategy=dictionary
```

## Credits & Contact Information

This implementation of PERDIF was written by Dimitris Berberidis and Athanasios N. Nikolakopoulos.

If you encounter any problems or have any suggestions, please contact Athanasios N. Nikolakopoulos at <a href="mailto:anikolak@umn.edu">anikolak@umn.edu</a>.


## Copyright & License Notice
Copyright 2019, Regents of the University of Minnesota

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
