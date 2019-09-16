# Personalized-Diffusions

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

SLIM uses Git submodules to manage external dependencies. Hence, please specify the `--recursive` option while cloning the repo as follow:
```bash
git clone https://github.com/nikolakopoulos/Personalized-Diffusions.git
```

## Building standalone PERDIF binary 

To build PERDIF you can follow the instructions below:

### Dependencies

General dependencies for building slim are: gcc, cmake, build-essential, blas.
In Ubuntu systems these can be obtained from the apt package manager (e.g., apt-get install cmake, etc) 

```bash
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libblas-dev
```

### Building and installing SLIM  

In order to build SLIM, run:

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

## Credits & Contact Information

This implementation of PERDIF was written by Dimitris Berberidis and Athanasios N. Nikolakopoulos.

If you encounter any problems or have any suggestions, please contact Athanasios N. Nikolakopoulos at <a href="mailto:anikolak@umn.edu">anikolak@umn.edu</a>.


## Copyright & License Notice
Copyright 2019, Regents of the University of Minnesota

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
