
ocrd <- function(X,beta,q0=0.5,eps=1e-10,iter.max = 10){
  n <- nrow(X)
  p_x <- rep(1/n,n)
  w <- apply(X*p_x,2,mean)
  w_old <- Inf
  iter <- 0
  while(sqrt(sum((w-w_old)^2) > eps & iter < iter.max)){
    iter <<- iter + 1
    w_old <- w
    
    d_x <- apply(X,1,function(v)sqrt(sum((v-w)^2)))
    i <- 0
    X_sort <- X[order(apply(X,1,function(v){
      i <<- i+1
      beta*d_x[i]+log2(p_x[i])
    })),]
    
    k <- n
    a_k <- 1
    p_k <- 1
    J_k <- beta*sum(p_x*d_x)
    k_best <- k
    J_best <- J_k
    
    while(k > 1){
      a_k1 <- a_k - 2^(-beta*d_x[k-1])
      p_k1 <- p_k - p_x[k-1]
      if (p_k1 <= a_k1){
        J_k <- J_k - p_x[k-1] * (beta*d_x[k-1]+log2(p_x[k-1])) + (p_k*log2(p_k) - p_k*log2(a_k)) -
          (p_k1*log2(p_k1) - p_k1*log2(a_k1))
        if (J_k < J_best){
          k_best <- k
          J_best <- J_k
        }
      }
      a_k <- a_k1
      p_k <- p_k1
      k <- k-1
    }
    q0x <- q0*2^(-beta*d_x)/p_x
    q0x[which(q0x > 1)] <- 1
    q0 <- sum(p_x[which(q0x == 1)]) + q0*sum(2^(-beta*d_x[which(q0x != 1)]))
    w <- colMeans(X*q0x)
    if (iter == iter.max){
      warning("iter.max reached without convergence")
    }
  }
  return(which(q0x!=1))
}

kmeansVanilla <- function(x,k,nstart,seed=NULL){
  if(!is.null(seed)){
    set.seed(seed)
  }
  kms <- kmeans(x,k,nstart = nstart)
  clustering <- kms$cluster
  freqs <- kms$size
  singletons <- which(freqs == 1)
  return(which(clustering %in% singletons))
}
kmeansMinusMinus <- function(X,k,l){
  n <- nrow(X)
  centers <- X[sample(1:n)[1:k],]
  i <- 1
  while (TRUE){
    dists <- t(apply(X,1,function(xi){
      apply(centers,1,function(ci){
        sqrt(sum((xi-ci)^2))
      })
    }))
    min_dists <- apply(dists,1,min)
    clustering <- apply(dists,1,which.min)
    outs <- order(min_dists,decreasing = TRUE)[1:l]
    
    drops <- which(unlist(lapply(1:k,function(j)all(which(clustering == j) %in% outs))))
    
    centers_new <- unname(unlist(lapply(setdiff(1:k,drops),function(j){
      idx <- setdiff(which(clustering == j),outs)
      if (length(idx) == 1){
        return(X[idx,])
      }
      colMeans(X[setdiff(which(clustering == j),outs),])
    })))
    k <- k - length(drops)
    
    stopifnot(length(centers_new) == k*ncol(X))
    centers_new <- matrix(unname(unlist(centers_new)),nrow = k,ncol=ncol(X),byrow = TRUE)

    if (length(drops) == 0 & sum(colSums(centers-centers_new)^2) < 1e-05){
      break
    }
    centers <- centers_new
    i <- i+1
  }
  return(outs)
}

kmor <- function(X,k,gamma,n_0,delta,iter_max){
  n <- nrow(X)
  Z <- X[sample(1:n)[1:k],]
  s <- 0
  p <- 0
  p_next <- Inf
  
  clustering <- rep(1,n)
  dists <- rep(Inf,n)
  
  while(TRUE){
    for (i in 1:n){
      d_average <- gamma * mean(dists[which(clustering != k+1)])
      for (j in 1:k){
        d <- sqrt(sum((X[i,] - Z[j,])^2))
        if (d < dists[i]){
          clustering[i] <- j
          dists[i] <- d
        }
        if (dists[i] > d_average){
          clustering[i] <- k+1
          dists[i] <- d_average
        }
      }
    }
    for (j in 1:k){
      Z[j,] <- colMeansSafe(X[which(clustering != k+1),])
    }
    s <- s+1
    p_next <- sum(dists)
    if (abs(p - p_next) < delta | s >= iter_max)
    {
      break
    }
    p <- p_next
  }
  return(which(clustering == k+1))
}

cblof <- function(X,b,theta,clustering){
  stopifnot(b < theta)
  
  compression <- clustering$cluster
  centers <- clustering$representation
  
  n <- nrow(X)
  cluster_sizes <- unname(sort(table(compression),decreasing = TRUE))
  cluster_order <- order(table(compression),decreasing = TRUE)
  
  large_clusters <- cluster_order[1:floor(b)]
  small_clusters <- cluster_order[floor(b+1):theta]
  i <- 0
  factor <- apply(X,1,function(xi){
    i <<- i+1
    if (compression[i] %in% large_clusters){
      return(cluster_sizes[compression[i]]*sqrt(sum((xi-centers[i,])^2)))
    }
    else{
      if (length(large_clusters) == 1){
        return(cluster_sizes[compression[i]] * sqrt(sum((xi-centers[large_clusters,])^2)))
      }
      dists <- apply(centers[large_clusters,],1,function(center){
        sqrt(sum((xi-center)^2))
      })
      return(cluster_sizes[compression[i]]*min(dists))
    }
  })
  return(factor)
}

cblofKnn <- function(X,b,min_pts,eps,dists,cluster){
  n <- nrow(X)
  
  
  freqs <- table(cluster)
  cluster_sizes <- unname(sort(table(cluster),decreasing = TRUE))
  cluster_order <- order(table(cluster),decreasing = TRUE)
  
  large_clusters <- cluster_order[1:floor(b)]
  small_clusters <- cluster_order[floor(b+1):length(freqs)]
  
  means <- unlist(lapply(unique(cluster),function(u){
    idx <- which(cluster == u)
    return(mean(dists[idx]))
  }))
  
  cands <- which(cluster %in% large_clusters)
  
  i <- 0
  factor <- apply(X,1,function(xi){
    i <<- i+1
    if (cluster[i] %in% large_clusters){
      return(cluster_sizes[cluster[i]] * (dists[i] - 1/freqs[cluster[i]] * means[cluster[i]]))
    }
    else{
      ds <- sqrt(rowSums(sweep(X,2,xi)^2))
      index <- which.min(ds[cands])
      return(cluster_sizes[cluster[i]]*(ds[cands[index]] - 1/freqs[cluster[cands[index]]] * means[cands[index]]))
    }
  })
  return(factor)
  
}

hacVanilla <- function(distances,ncluster){
  require(fastcluster)
  
  cluster <- cutree(fastcluster::hclust(distances,method = "complete"),k = ncluster)
  u <- unique(cluster)
  o <- integer(0)
  for (i in 1:length(u)){
    idx <- which(cluster == i)
    if (length(idx) == 1)
    {
      o <- c(o,idx)
    }
  }
  return(o)
}

dbscanVanilla <- function(x,minPts,eps){
  require(dbscan)
  return(which(dbscan::dbscan(x,minPts = minPts,eps=eps)$cluster == 0))
}

lps <- function(x,k,to,knn=NULL){
  x <- scale(x)
  require(dbscan)
  n <- nrow(x)
  r <- length(svd(x)$d)
  if (to > r){
    to <- r
  }
  if (is.null(knn)){
    print("computing knn")
    knn <- dbscan::kNN(x,k,sort = FALSE)$id
  }
  # knn <- x[knn[,1:k],]
  # result <- apply(knn,1,function(neighbors){
  # 
  #   singular <- svd(neighbors,nu=to,nv=to)
  #   rho <- min(c(length(singular$d),to))
  #   D <- singular$u[1:rho,] %*% (diag(rho)*singular$d[1:rho]) %*% t(singular$v[1:rho,])
  #   return(sum(svd(D)$d))
  # })
  result <- rep(NaN,n)
  for (i in 1:n){
    neighbors <- x[knn[i,1:k],]
    singular <- svd(neighbors)
    D <- t(singular$u[1:to,]) %*% (diag(to)*singular$d[1:to]) %*% singular$v[1:to,]
    result[i] <- sum(svd(D)$d)
  }
  return(result)
}