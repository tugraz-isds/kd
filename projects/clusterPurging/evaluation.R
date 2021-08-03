evalMetrics <- function(predicted,actual,n,beta=1,rev_only=FALSE){
  require(pbapply)
  m <- length(actual)
  tp <- sum(predicted %in% actual)
  if (tp == 0)
  {
    return(c(precision=0,recall=0,f_score=0,mcc=0))
  }
  fn <- (m - tp)
  fp <- length(predicted) - tp
  tn <- n-m-fp
  
  if (rev_only){
    return(c(fpr=fp/(fp+tn),tpr=tp/(tp+fn)))
  }
  
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  f <- (1+beta^2) * precision * recall / (beta^2 * precision + recall)
  if (tn == 0)
  {
    return(c(precision=precision,recall=recall,f_score=f,mcc=0))
  }
  mcc <- (tp*tn - fp*fn)/sqrt(as.numeric(tp+fp)*as.numeric(tp+fn)*as.numeric(tn+fp)*as.numeric(tn+fn))
  return(c(precision=precision,recall=recall,f_score=f,mcc=mcc))
}


caseStudy <- function(noise=matrix(nrow=0,ncol=2)){
  europe <- as.matrix(read.table('europe/europe.txt',sep=' ')[1:2])
  

  set.seed(0815)
 
  if (nrow(noise)==0){
    n1 <- c(-20,40)
    n2 <- c(-10,80)
    n3 <- c( 60,41)
    n4 <- c( 70,60)
    noise <- rbind(n1,n2,n3,n4,noise)
    while (nrow(noise) < 100){
      print(nrow(noise))
      ni <- c(runif(1,-40,80),runif(1,25,90))
      d_europe <- min(apply(europe,1,function(p)sqrt(sum((p-ni)^2))))
      d_noise <- min(apply(noise,1,function(p)sqrt(sum((p-ni)^2))))
      
      if (d_europe < 3.5 | d_noise < 3.5){
        next
      }
      noise <- rbind(noise,ni)
    }
    return(noise)
  }
  
  y1 <- nrow(europe) + 1:length(noise)
  y2 <- 1:nrow(europe)
  
  europe <- rbind(europe,noise)
  n <- nrow(europe)
  
  res1 <- matrix(NA,nrow = 10,ncol=4)
  res2 <- matrix(NA,nrow = 10,ncol=4)
  colnames(res1) <- c("MinMin","MinMax","MaxMin","MaxMax")
  colnames(res2) <- c("MinMin","MinMax","MaxMin","MaxMax")
  
  for (i in 1:10){
    cat(paste0("Iteration ",i,"\n"))
    set.seed(0815 + i)
    cmp1 <- kmeansCompression(europe,225,nstart = 1000)
    cmp1.1 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMinMin(europe,cmp1))
    cmp1.2 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMinMax(europe,cmp1))
    cmp1.3 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMaxMin(europe,cmp1))
    cmp1.4 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMaxMax(europe,cmp1))
    
    o1 <- clusterPurging(europe,list(cmp1,cmp1.1))
    o2 <- clusterPurging(europe,list(cmp1,cmp1.2))
    o3 <- clusterPurging(europe,list(cmp1,cmp1.3))
    o4 <- clusterPurging(europe,list(cmp1,cmp1.4))
    
    res1[i,1] <- evalMetrics(o1,y1,n)[3]
    res1[i,2] <- evalMetrics(o2,y1,n)[3]
    res1[i,3] <- evalMetrics(o3,y1,n)[3]
    res1[i,4] <- evalMetrics(o4,y1,n)[3]
    
    res2[i,1] <- evalMetrics(which(!(1:n %in% o1)),y2,n)[3]
    res2[i,2] <- evalMetrics(which(!(1:n %in% o2)),y2,n)[3]
    res2[i,3] <- evalMetrics(which(!(1:n %in% o3)),y2,n)[3]
    res2[i,4] <- evalMetrics(which(!(1:n %in% o4)),y2,n)[3]
  }
  print(round(colMeans(res1),2))
  print(round(colMeans(res2),2))
  print(round(colMeans(1/2 * (res1 + res2)),2))
  return()
  
}

competitiveEvaluation <- function(testPerturb=FALSE){
  source('gridSearchers.R')
  if (testPerturb){
    gridSearchers <- c(KMminmin=function(X,y)cpKmeansGridSearch(X,y,perturbFunc = perturbationMinMin),
                       KMminmax=function(X,y)cpKmeansGridSearch(X,y,perturbFunc = perturbationMinMax),
                       KMmaxmin=function(X,y)cpKmeansGridSearch(X,y,perturbFunc = perturbationMaxMin),
                       KMmaxmax=function(X,y)cpKmeansGridSearch(X,y,perturbFunc = perturbationMaxMax),
                       HACminmin=function(X,y)cpHacGridSearch(X,y,perturbFunc = perturbationMinMin),
                       HACminmax=function(X,y)cpHacGridSearch(X,y,perturbFunc = perturbationMinMax),
                       HACmaxmin=function(X,y)cpHacGridSearch(X,y,perturbFunc = perturbationMaxMin),
                       HACmaxmax=function(X,y)cpHacGridSearch(X,y,perturbFunc = perturbationMaxMax),
                       DBminmin=function(X,y)cpDbscanGridSearch(X,y,perturbFunc = perturbationMinMin),
                       DBminmax=function(X,y)cpDbscanGridSearch(X,y,perturbFunc = perturbationMinMax),
                       DBmaxmin=function(X,y)cpDbscanGridSearch(X,y,perturbFunc = perturbationMaxMin),
                       DBmaxmax=function(X,y)cpDbscanGridSearch(X,y,perturbFunc = perturbationMaxMax))
  }
  else{
    gridSearchers <- c(ocrd=ocrdGridSearch,
                      vanillakm = kmeansVanillaGridSearch,kmmm=kmeansMinusMinusGridSearch,
                      kmor=kmorGridSearch,cblofkm=cblofKmeansGridSearch,cpkm=cpKmeansGridSearch,cppkm=cppKmeansGridSearch,
                      vanillahac=hacVanillaGridSearch,cblofhac=cblofHacGridSearch,cphac=cpHacGridSearch,cpphac=cppHacGridSearch,
                      vanilladb=dbscanVanillaGridSearch,cblofdb=cblofDbscanGridSearch,cpdb=cpDbscanGridSearch,cppdb=cppDbscanGridSearch,
                      lof=lofGridSearch,lps=lpsGridSearch)
  }
 
   
  
  

  m <- length(gridSearchers)
  
  datasets <- list()
  
  datasets[[1]] <- processArff("semantic/nondupl/Arrhythmia_withoutdupl_46.arff")
  datasets[[2]] <- processArff("semantic/nondupl/HeartDisease_withoutdupl_44.arff")
  datasets[[3]] <- processArff("semantic/nondupl/Hepatitis_withoutdupl_16.arff")
  datasets[[4]] <- processArff("semantic/nondupl/Parkinson_withoutdupl_75.arff")
  datasets[[5]] <- processArff("semantic/nondupl/Pima_withoutdupl_35.arff")
  datasets[[6]] <- processArff("semantic/nondupl/Stamps_withoutdupl_09.arff")
  datasets[[7]] <- processArff("non_semantic/Glass_withoutdupl_norm.arff")
  datasets[[8]] <- processArff("non_semantic/Ionosphere_withoutdupl_norm.arff")
  datasets[[9]] <- processArff("non_semantic/Lymphography_withoutdupl_idf.arff")
  datasets[[10]] <- processArff("non_semantic/Shuttle_withoutdupl_v01.arff")
  datasets[[11]] <- processArff("non_semantic/WBC_withoutdupl_v01.arff")
  datasets[[12]] <- processArff("non_semantic/WDBC_withoutdupl_v01.arff")
  datasets[[13]] <- processArff("non_semantic/WPBC_withoutdupl_norm.arff")
  
  
  results <- matrix(NaN,nrow = length(datasets),ncol=m*5)
  colnames(results) <- rep(names(gridSearchers),5)
  rownames(results) <- c("Arrhymthia","Heart","Hepatitis","Parkinson","Pima","Stamps",
                         "Glass","Ionosphere","Lympho","Shuttle","WBC","WDBC","WPBC")
  i <- 0
  for (dataset in datasets){
    i <- i + 1
    X <- dataset$X / max(abs(dataset$X))
    Y <- which(dataset$y==1)
    j <- 0
    for (gridSearcher in gridSearchers){
      j <- j+1
      print(names(gridSearchers)[j])
      start_time <- Sys.time()
      l <- gridSearcher(as.matrix(X),Y)
      predictions <- l$predictions
      iter <- l$iter
      end_time <- Sys.time()
      performance <- c(precision=evalMetrics(predictions[[1]],Y,nrow(X))[["precision"]],
                       recall=evalMetrics(predictions[[2]],Y,nrow(X))[["recall"]],
                       f_score=evalMetrics(predictions[[3]],Y,nrow(X))[["f_score"]],
                       mcc=evalMetrics(predictions[[4]],Y,nrow(X))[["mcc"]],
                       time=as.numeric(difftime(end_time,start_time,unit="secs"))/iter)
      print(performance)
      results[i,(1:5)+(m-1)*(0:4)+j-1] <- performance
      
    }
    
  }
  
  return(list(precision=results[,1:m],recall=results[,(m+1):(2*m)],f_score=results[,(2*m+1):(3*m)],
              mcc=results[,(3*m+1):(4*m)],time=results[,(4*m+1):(5*m)]))
}


processArff <- function(file){
  require(farff)
  dataset <- farff::readARFF(file,convert.to.logicals = FALSE)
  return(list(X=as.matrix(dataset[,1:(ncol(dataset)-2)]),y=dataset$outlier=='yes'))
}

