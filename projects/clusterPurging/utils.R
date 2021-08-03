kmeansCompression <- function(x,k,nstart=100,cluster=NULL){
  if (is.null(cluster)){
    kms <- kmeans(x,k,nstart = nstart)
    return(list(cluster=kms$cluster,representation=kms$centers[kms$cluster,]))
  }
  return(list(cluster=cluster,representation=meanCenters(x,cluster)))
  
}

hacCompression <- function(x,n_components,method='complete',dists=NULL,cluster=NULL,hclust=TRUE){
  if (is.null(cluster)){
    if (is.null(dists)){
      dists <- dist(x)
    }
    cluster <- unname(cutree(fastcluster::hclust(dists,method=method),k = n_components))
  }
  if (!hclust){
    cluster <- unname(cutree(cluster,k=n_components))
  }
  return(list(cluster=cluster,representation=meanCenters(x,cluster)))
}

dbscanCompression <- function(x,min_pts, eps,cluster=NULL,dists=NULL){
  dists <- dbscan::frNN(x,eps)
  if (is.null(cluster)){
    cluster <- dbscan::dbscan(dists,minPts = min_pts,eps = eps)$cluster
  }
  
  o <- which(cluster == 0)
  cluster[o] <- max(cluster) + 1:length(o)
  repres <- x
  for (j in 1:nrow(x)){
    if (length(cluster[dists$id[[j]]]) == 0){
      repres[j,] <- x[j,]
      next
    }
    neighbors <- dists$id[[j]]
    in_same_cluster <- which(cluster[neighbors]==cluster[j])
    if (length(in_same_cluster) == 0){
      repres[j,] <- x[j,]
    }
    else{
      repres[j,] <- x[neighbors[in_same_cluster[1]],]
    }
  }
  return(list(cluster=cluster,representation=repres))
}


colMeansSafe <- function(X,...){
  if (is.null(dim(X))){
    return(mean(X))
  }
  return(colMeans(X,...))
}

meanCenters <- function(x,cluster){
  uniques <- unique(cluster)
  v <- length(uniques)
  d <- ncol(x)
  if (v == 1){
    n <- length(cluster)
    return(matrix(rep(colMeansSafe(x),n),nrow=n,byrow = TRUE))
  }
  centers <- matrix(nrow=v,ncol=d)
  for (u in 1:v){
    centers[u,] <- colMeansSafe(matrix(x[which(cluster==uniques[u]),],ncol=d))
  }
  
  return(centers[match(cluster,uniques),])
}

distortionEuclid <- function(x,R){
  if (is.null(nrow(x))){
    return(sqrt(sum((x-R)^2)))
  }
  s <- 0
  for (i in 1:nrow(x))
  {
    s <- s + sqrt(sum((x[i,] - R[i,])^2))
  }
  return(s)
}
distortionPerCluster <- function(x,clustering,distortion=distortionEuclid){
  u <- unique(clustering$cluster)
  return(unlist(sapply(1:length(u),function(cl){
    idx <- which(clustering$cluster == cl)
    if (length(idx) == 1){
      return(0)
    }
    return(distortionEuclid(x[idx,],clustering$representation[idx,]))
  })))
}
perturbationMaxMax <- function(x,clustering,distortion=distortionEuclid){
  target_cluster <- which.max(tabulate(clustering$cluster))
  idx <- which(clustering$cluster == target_cluster)
  dists <- unlist(lapply(idx,function(i)distortion(x[i,],clustering$representation[i,])))
  target <- which.max(dists)
  new_cluster <- clustering$cluster
  new_cluster[idx[target]] <- length(unique(new_cluster)) + 1
  return(new_cluster)
}

perturbationMaxMin <- function(x,clustering,distortion=distortionEuclid){
  #freqs <- tabulate(clustering$cluster)
  #target_cluster <- which(freqs > 1)[which.min(freqs[which(freqs > 1)])]
  target_cluster <- which.max(tabulate(clustering$cluster))
  idx <- which(clustering$cluster == target_cluster)
  dists <- unlist(lapply(idx,function(i)distortion(x[i,],clustering$representation[i,])))
  target <- which.min(dists)
  new_cluster <- clustering$cluster
  new_cluster[idx[target]] <- length(unique(new_cluster)) + 1
  
  return(new_cluster)
}

perturbationMinMax <- function(x,clustering,distortion=distortionEuclid){
  freqs <- tabulate(clustering$cluster)
  target_cluster <- which(freqs > 1)[which.min(freqs[which(freqs > 1)])]
  #target_cluster <- which.max(tabulate(clustering$cluster))
  idx <- which(clustering$cluster == target_cluster)
  dists <- unlist(lapply(idx,function(i)distortion(x[i,],clustering$representation[i,])))
  target <- which.max(dists)
  new_cluster <- clustering$cluster
  new_cluster[idx[target]] <- length(unique(new_cluster)) + 1
  
  return(new_cluster)
}

perturbationMinMin <- function(x,clustering,distortion=distortionEuclid){
  freqs <- tabulate(clustering$cluster)
  target_cluster <- which(freqs > 1)[which.min(freqs[which(freqs > 1)])]
  #target_cluster <- which.max(tabulate(clustering$cluster))
  idx <- which(clustering$cluster == target_cluster)
  dists <- unlist(lapply(idx,function(i)distortion(x[i,],clustering$representation[i,])))
  target <- which.min(dists)
  new_cluster <- clustering$cluster
  new_cluster[idx[target]] <- length(unique(new_cluster)) + 1
  
  return(new_cluster)
}

lhull <- function(x,y){
  hull <- chull(x,y)
  lower_hull <- rev(hull[1:which(hull == hull[which.min(x[hull])])])
  curve <- lower_hull[c(1,which(diff(y[lower_hull]) < 0)+1)]
  x_curve <- x[curve]
  y_curve <- y[curve]
  s <- length(curve)
  return(list(curve=matrix(c(x_curve,y_curve),ncol=2),indices=curve))
  
}
changeOfEntropy <- function(x,n){
  ns <- which(x > 1)
  result <- rep(0,length(x))
  result[ns] <- (x[ns]*log2(x[ns]) - (x[ns]-1) * log2(x[ns]-1)) / n
  return(result)
}