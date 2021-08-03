This folder contains instructions an R implementation of Cluster Purging, as well as code and datasets used for the experiments in the related paper.

If you use any of our code in your paper, please use add the following citation:

```
@article{toller2021cluster,
  title={Cluster Purging: Efficient Outlier Detection based on Rate-Distortion Theory},
  author={Toller, Maximilian Bernhard and Geiger, Bernhard Claus and Kern, Roman},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={TBA},
  number={TBA},
  pages={TBA},
  year={TBA},
  publisher={IEEE}
}
```

All code is written in R, methods that were only available in other languages were reimplemented in R.
A Python implementation will be added in the near future.

-type "source('compile.R')" in the R console to load all code.
-clusterPurging.R contains the implementations of our proposed algorithms.
-evaluation.R contains the main source code of our case study and the competitive evaluation.
-plots.R contains methods for creating all plots in the paper.
-utils.R contains utility functions such as euclidean distances, and perturbation methods.
-competitors.R contains R source code for related methods.
-gridSearchers.R contains code for performing the grid searches.
 

You can also test our implementation on any further dataset you wish by simply calling "clusterPurging(X,ListOfClusterings)",
where X is the dataset in matrix form and ListOfClusterings is a list of cluster-indices - representation pairs, cf. utils.R

The required R packages and versions are listed here:
MultiRNG_1.2.2      mvtnorm_1.1-0       spatstat_1.64-1     rpart_4.1-15        nlme_3.1-147
spatstat.data_1.4-3 dbscan_1.1-5        solitude_0.2.1      fastcluster_1.1.25  Rlof_1.1.1
doParallel_1.0.15   iterators_1.0.12    foreach_1.4.8       pbapply_1.4-2       readr_1.3.1
farff_1.1
