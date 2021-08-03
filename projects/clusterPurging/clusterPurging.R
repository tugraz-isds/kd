clusterPurging <- function(x,clusterings,pl=FALSE,...){
  n <- nrow(x)
  
  freqss <- lapply(clusterings,function(clustering)table(clustering$cluster))

  centerss <- lapply(clusterings,function(clustering)clustering$representation)
  es <- unlist(lapply(freqss,function(freqs)log2(n) - 1/n*sum(freqs*log2(freqs))))
  ds <- unlist(lapply(centerss,function(centers)distortionEuclid(x,centers)))
  
  #print(es)
  #print(ds)
  
  lower_hull <- lhull(ds,es)
  if (nrow(lower_hull$curve) < 2){
    warning("Rate-distortion curve has length < 2")
    return(integer(0))
  }
  rd_curve <- lower_hull$curve
  indices <- lower_hull$indices

  # Construct RD curve
  s <- nrow(rd_curve)
  k <- rep(NaN,s+1)
  for (i in 2:s){
    k[i] <- (rd_curve[i,2] - rd_curve[i-1,2]) / (rd_curve[i,1]-rd_curve[i-1,1])
  }
  k[1] <- k[2]
  k[s+1] <- k[s]
  
  
  purging_boundaries <- lapply(indices,function(i)changeOfEntropy(freqss[[i]],n)/-k[which(indices == i)])
  o <- matrix(TRUE,nrow = n,ncol=s)
  for (j in 1:n){
    for (i in 2:(s)){
      target_cluster <- clusterings[[indices[i]]]$cluster[j]
      purging_boundary <- purging_boundaries[[i]][target_cluster]
      o[j,i] <- sqrt(sum((x[j,]-centerss[[indices[i]]][j,])^2)) >= purging_boundary
      if (!o[j,i]){
        {
          break
        }
      }
    }
  }
  if (pl){
    List <- list()
    for (i in 2:s){
      List[[i]] <- list()
      List[[i]]$centers <- centerss[[indices[i]]][which(!duplicated(centerss[[indices[i]]])),]
      List[[i]]$radii <- purging_boundaries[[i]][clusterings[[indices[i]]]$cluster[which(!duplicated(centerss[[indices[i]]]))]]
      List[[i]]$outliers <- which(o[,i])
    }
    createAreaPlotMult(x,List,which(apply(o,1,all)),...)
  }
  return(which(apply(o,1,all)))
}

clusterPurgingParametric <- function(x,clustering,k,pl=FALSE,...){
  n <- nrow(x)
  
  d <- distortionEuclid(x,clustering$representation)
  
  freqs <- table(clustering$cluster)
  
  ce <- changeOfEntropy(freqs,n)
  
  r <- ce/k*d
  
  o <- rep(FALSE,n)
  for (j in 1:n){
    if (sqrt(sum((x[j,]-clustering$representation[j,])^2)) >= r[clustering$cluster[j]]){
      o[j] <- TRUE
    }
  }
  if (pl){
    l <- list()
    l$centers <- clustering$representation[which(!duplicated(clustering$representation)),]
    l$radii <- r[clustering$cluster[which(!duplicated(clustering$representation))]]
    l$outliers <- which(o)
    createAreaPlotMult(x,list(l),which(o),...)
  }
  return(which(o))
  
}