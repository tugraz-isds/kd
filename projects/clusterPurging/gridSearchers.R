ocrdGridSearch <- function(X,y){
  require(pbapply)
  X <- scale(X)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  results <- matrix(unlist(pblapply(seq(0.1,10,length.out = n),function(beta){
    iter <<- iter + 1
    o[[iter]] <<- ocrd(X,beta)
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

kmeansVanillaGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    iter <<- iter + 1
    o[[iter]] <<- kmeansVanilla(X,k,1000,seed = 0815)
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter*1000))
}

kmeansMinusMinusGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  set.seed(0815)
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    iter <<- iter + 1
    o[[iter]] <<- kmeansMinusMinus(X,k,m)
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

kmorGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  set.seed(0815)
  gs <- seq(0.1,10,length.out = 30)
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    return(unlist(lapply(gs,function(g){
      iter <<- iter + 1
      o[[iter]] <<- kmor(X,k,g,n_0 = 0,delta = 1,iter_max = 10)
      return(evalMetrics(o[[iter]],y,n))
    })))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cblofKmeansGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    iter <<- iter + 1
    set.seed(seed = 0815)
    b <- min(c(k-1,5))
    o[[iter]] <<- rev(order(cblof(X,b,k,clustering = kmeansCompression(X,k,nstart = 1000))))[1:m]
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter*1000))
}

cpKmeansGridSearch <- function(X,y,perturbFunc=perturbationMaxMax){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    iter <<- iter + 1
    set.seed(seed = 0815)
    cmp <- kmeansCompression(X,k,nstart = 1000)
    
    perturbation <- kmeansCompression(X,-1,nstart = 0,cluster = perturbFunc(X,cmp))
    o[[iter]] <<- clusterPurging(X,ncluster = NaN,clustering = list(cmp,perturbation))
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter*1000))
}
cppKmeansGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  ks <- seq(0.1,10,length.out = 30)
  
  results <- matrix(unlist(pblapply(rev(2:10),function(k){
    set.seed(seed = 0815)
    clustering <- kmeansCompression(X,k,nstart = 1000)
    return(unlist(lapply(ks,function(k){
      iter <<- iter + 1
      res <- clusterPurgingParametric(X,clustering = clustering,k=k)
      if (is.null(res)){
        res <- list()
      }
      o[[iter]] <<- res
      return(evalMetrics(o[[iter]],y,n))
    })))
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter + (iter-9*29)*1000))
}

hacVanillaGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  distances <- dist(X,method = 'euclidean')
  
  results <- matrix(unlist(pblapply(rev(1:(n)),function(theta){
    iter <<- iter + 1
    o[[iter]] <<- hacVanilla(distances,theta)
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cblofHacGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  distances <- dist(X,method = 'euclidean')
  
  
  results <- matrix(unlist(pblapply(3:(n-1),function(theta){
    iter <<- iter + 1
    b <- min(c(theta-1,5))
    clustering <- hacCompression(X,theta,dists = distances)
    o[[iter]] <<- rev(order(cblof(X,b,theta,clustering = clustering)))[1:m]
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cpHacGridSearch <- function(X,y,perturbFunc=perturbationMaxMax){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  distances <- dist(X)
  clustering <- fastcluster::hclust(distances,method = "complete")
  results <- matrix(unlist(pblapply(rev(1:(n-1)),function(theta){
    
    iter <<- iter + 1
    cmp <- hacCompression(X,theta,cluster = clustering,hclust = FALSE)
    pertubation <- hacCompression(X,-1,cluster = perturbFunc(X,cmp),hclust=TRUE)
    res <- clusterPurging(X,list(cmp,pertubation))
    if (is.null(res)){
      res <- list()
    }
    o[[iter]] <<- res
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cppHacGridSearch <- function(X,y,rev_only=FALSE){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  iter <- 0
  
  distances <- dist(X)
  clustering <- fastcluster::hclust(distances,method = "complete")

  if (rev_only){
    ks <- seq(0.1,100,length.out = 3000)
  }
  else{
    ks <- seq(0.1,10,length.out = 30)
  }
  #ks <- ifelse(rev_only,3,seq(0.1,10,length.out = 30))
  thetas <- ifelse(rev_only,5,rev(1:(n-1)))
  results <- matrix(unlist(pblapply(thetas,function(theta){
    return(unlist(lapply(ks,function(k){
      iter <<- iter + 1
      cmp <- hacCompression(X,theta,cluster = clustering,hclust = FALSE)
      res <- clusterPurgingParametric(X,clustering = cmp,k=k)
      if (is.null(res)){
        res <- list()
      }
      o[[iter]] <<- res
      
      if (rev_only){
        return(evalMetrics(o[[iter]],y,n,rev_only=TRUE))
      }
      
      return(evalMetrics(o[[iter]],y,n))
    })))
  })),nrow=ifelse(rev_only,2,4))
  
  if (rev_only){
    return(results)
  }
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

dbscanVanillaGridSearch <- function(X,y){
  require(pbapply)
  require(dbscan)
  
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  
  iter <- 0
  kappa <- ncol(X)+1:10
  results <- matrix(unlist(pblapply(kappa,function(minPts){
    epss <- sort(unique(dbscan::kNNdist(X,minPts,all=TRUE)[,minPts]))
    
    return(unlist(lapply(epss,function(eps){
      iter <<- iter + 1
      o[[iter]] <<- dbscanVanilla(X,minPts,eps)
      return(evalMetrics(o[[iter]],y,n))
    })))
    
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cblofDbscanGridSearch <- function(X,y){
  require(pbapply)
  require(dbscan)
  
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  dists <- dbscan::kNNdist(X,1)
  iter <- 0
  kappa <- ncol(X)+1:10
  results <- matrix(unlist(pblapply(kappa,function(minPts){
    epss <- sort(unique(dbscan::kNNdist(X,minPts,all=TRUE)[,minPts]))
    
    return(unlist(lapply(epss,function(eps){
      iter <<- iter + 1
      clustering <- dbscanCompression(X,minPts,eps)$cluster
      b <- min(c(length(unique(clustering))-1,5))
      o[[iter]] <<- rev(order(cblofKnn(X,b,min_pts = minPts,eps = eps,cluster = clustering,dists = dists)))[1:m]
      return(evalMetrics(o[[iter]],y,n))
    })))
    
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cpDbscanGridSearch <- function(X,y,perturbFunc=perturbationMaxMax){
  require(pbapply)
  require(dbscan)
  
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  
  iter <- 0
  kappa <- ncol(X)+1:10
  results <- matrix(unlist(pblapply(kappa,function(minPts){
    epss <- sort(unique(dbscan::kNNdist(X,minPts,all=TRUE)[,minPts]))
    
    return(unlist(lapply(epss,function(eps){
      iter <<- iter + 1
      cmp <- dbscanCompression(X,minPts,eps)
      perturbation <- dbscanCompression(X,minPts,eps,cluster = perturbFunc(X,cmp))
      o[[iter]] <<- clusterPurging(X,list(cmp,perturbation))
      return(evalMetrics(o[[iter]],y,n))
    })))
    
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

cppDbscanGridSearch <- function(X,y){
  require(pbapply)
  require(dbscan)
  
  n <- nrow(X)
  m <- length(y)
  
  o <- list()
  
  iter <- 0
  kappa <- ncol(X)+1:10
  ks <- seq(0.1,10,length.out = 30)
  results <- matrix(unlist(pblapply(kappa,function(minPts){
    epss <- sort(unique(dbscan::kNNdist(X,minPts,all=TRUE)[,minPts]))
    
    return(unlist(lapply(epss,function(eps){
      clustering <- dbscanCompression(X,minPts,eps)
      return(unlist(lapply(ks,function(k){
        iter <<- iter + 1
        res <- clusterPurgingParametric(X,clustering,k)
        if (is.null(res)){
          res <- list()
        }
        o[[iter]] <<- res
        return(evalMetrics(o[[iter]],y,n))
      })))
    })))
    
  })),nrow=4)
  
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

lpsGridSearch <- function(X,y){
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  d <- ncol(X)
  
  o <- list()
  
  ks <- 2:ceiling(d/2)
  knn <- dbscan::kNN(X,max(ks),sort = TRUE)$id
  iter <- 0
  results <- matrix(unlist(pblapply(rev(ks),function(k){
    iter <<- iter + 1
    o[[iter]] <<- rev(order(lps(X,k,to = 2,knn = knn)))[1:m]
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}

lofGridSearch <- function(X,y){
  require(Rlof)
  require(pbapply)
  n <- nrow(X)
  m <- length(y)
  d <- ncol(X)
  
  o <- list()
  
  ks <- 1:(n-1)

  iter <- 0
  results <- matrix(unlist(pblapply(rev(ks),function(k){
    iter <<- iter + 1
    o[[iter]] <<- rev(order(Rlof::lof(X,k)))[1:m]
    return(evalMetrics(o[[iter]],y,n))
  })),nrow=4)
  print(apply(results,1,max))
  return(list(predictions=o[apply(results,1,which.max)],iter=iter))
}