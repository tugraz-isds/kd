savepdf <- function(file, width=16, height=10){
  fname <- paste(file,".pdf",sep="")
  cairo_pdf(fname, width=width/2.54, height=height/2.54,
            pointsize=10)
  par(mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(3.3,3.6,1.1,1.1))
}

createAreaPlotMult <- function(X,List,fullOutliers,add=FALSE,xlab='',ylab='',cents=FALSE,mask=FALSE,cols=NULL,onlyZone=FALSE,noOutliers=FALSE,...){
  require(spatstat)
  if (!add){
    dev.new()
  }
  if (!onlyZone){
    plot(X,col='black',asp=1,xlab=xlab,ylab=ylab,...)
  }
  xran <- max(abs(range(X[,1])))
  yran <- max(abs(range(X[,2])))
  m <- length(List)
  iter <- 0
  if (is.null(cols)){
    cols <- c('darkgreen','blue','cyan','magenta','orange','gray',topo.colors(length(List)-6))
  }
  lapply(rev(List),function(l){
    iter <<- iter+1
    centers <- l$centers
    radii <- l$radii
    outliers <- l$outliers
    m2 <- length(radii)
    kappa <- which(radii>0)
    p <- ppp(x=centers[kappa,1],y = centers[kappa,2],window =
               owin(xrange = 2*c(-xran,xran),yrange=2*c(-yran,yran)))
    d <- discs(p,radii[kappa],mask = mask)
    plot(d,add=TRUE,border=cols[iter])
    if (cents){
      points(centers[,1],centers[,2],pch=2,col='magenta',lwd=3)
    }
    if (!noOutliers){
      for (j in outliers){
        points(X[j,1],X[j,2],col=cols[iter],pch=4,lwd=2)
      }
    }
    
  })
  if (!noOutliers){
    points(X[fullOutliers,1],X[fullOutliers,2],col='red',pch=4,lwd=4)
  }
  
}

circleNoise <- function(n){
  x <- rnorm(n)
  y <- runif(n)
  t <- 2*pi*x
  return(matrix(c(y*cos(t),y*sin(t)),ncol=2))
}
clusterToSymbolPlot <- function(){
  require(latex2exp)
  set.seed(0815)
  c1 <- sweep(circleNoise(30)*0.28,2,c(1,2),FUN="+")
  c2 <- sweep(circleNoise(20)*0.26,2,c(3,1),FUN="+")
  c3 <- sweep(circleNoise(15)*0.25,2,c(3.5,3),FUN="+")
  
  data <- rbind(c1,c2,c3)
  kms <- kmeans(data,data[c(1,31,51),])
  savepdf("clusterToSymbols",width = 12,height = 6)
  #par(mfrow=c(1,2))
  
  layout(t(c(1,4,2,5,3)),c(0.3,0.05,0.3,0.05,0.3))
  par(mar=c(0.1,0.1,1.5,1.5))
  
  plot(rbind(c1,c2,c3),xlim=c(0,5),ylim=c(0,4.5),pch=20,col='black',xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
  lines(c(0,0,5),c(5,0,0),col='black',lwd=2)
  lines(c(0,0.3),c(1,1))
  lines(c(0,0.3),c(2,2))
  lines(c(0,0.3),c(3,3))
  lines(c(0,0.3),c(4,4))
  lines(c(1,1),c(0,0.15))
  lines(c(2,2),c(0,0.15))
  lines(c(3,3),c(0,0.15))
  lines(c(4,4),c(0,0.15))
  text("Raw Data",x=0.3,y=4.4,pos = 4,cex=1.75,col=rgb(0.3,0.1,0.05))
  
  plot(c1,xlim=c(0,5),ylim=c(0,4.5),pch=20,col='blue',xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
  points(c2,pch=20,col='green')
  points(c3,pch=20,col='orange')
  lines(c(0,0,5),c(5,0,0),col='black',lwd=2)
  lines(c(0,0.3),c(1,1))
  lines(c(0,0.3),c(2,2))
  lines(c(0,0.3),c(3,3))
  lines(c(0,0.3),c(4,4))
  lines(c(1,1),c(0,0.15))
  lines(c(2,2),c(0,0.15))
  lines(c(3,3),c(0,0.15))
  lines(c(4,4),c(0,0.15))
  # arrows(kms$centers[1,1],kms$centers[1,2]+1,kms$centers[1,1],kms$centers[1,2]+0.2,lty=1,length = 0.1)
  # arrows(kms$centers[2,1],kms$centers[2,2]+1,kms$centers[2,1],kms$centers[2,2]+0.2,lty=1,length = 0.1)
  # arrows(kms$centers[3,1],kms$centers[3,2]+1,kms$centers[3,1],kms$centers[3,2]+0.2,lty=1,length = 0.1)
  text(TeX("$c = 2$"),x =kms$centers[1,1]+0.5 ,y=kms$centers[1,2]+0.7,cex = 2.05)
  text(TeX("$c = 3$"),x =kms$centers[2,1]-0.2 ,y=kms$centers[2,2]+0.7,cex = 2.05)
  text(TeX("$c = 1$"),x =kms$centers[3,1]+0.2 ,y=kms$centers[3,2]+0.7,cex = 2.05)
  text("Cluster Indices",x=0.3,y=4.4,pos = 4,cex=1.75,col=rgb(0.3,0.1,0.05))
  
  plot(kms$centers,xlim=c(0,5),ylim=c(0,4.5),col=c('blue','green','orange'),xaxt='n',yaxt='n',xlab='',ylab='',bty='n',pch=c(2,3,4),lwd=2)
  lines(c(0,0,5),c(5,0,0),col='black',lwd=2)
  lines(c(0,0.3),c(1,1))
  lines(c(0,0.3),c(2,2))
  lines(c(0,0.3),c(3,3))
  lines(c(0,0.3),c(4,4))
  lines(c(1,1),c(0,0.15))
  lines(c(2,2),c(0,0.15))
  lines(c(3,3),c(0,0.15))
  lines(c(4,4),c(0,0.15))
  text(TeX("$r_2$"),x =kms$centers[1,1]+0.5 ,y=kms$centers[1,2]+0.3,cex = 2.05)
  text(TeX("$r_3$"),x =kms$centers[2,1]-0.2 ,y=kms$centers[2,2]+0.3,cex = 2.05)
  text(TeX("$r_1$"),x =kms$centers[3,1]+0.2 ,y=kms$centers[3,2]+0.3,cex = 2.05)
  text("Representation",x=0.3,y=4.4,pos = 4,cex=1.75,col=rgb(0.3,0.1,0.05))
  plotfunctions::drawDevArrows(c(1.2,1.2),c(1.65,1.2),lwd=2,lty=1,col='darkgray')
  plotfunctions::drawDevArrows(c(2.8,1.2),c(3.25,1.2),lwd=2,lty=1,col='darkgray')
  dev.off()
  #plot(0,xlim=c(0,0.1),ylim=c(0,0.1),type='n',xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
  #arrows(3,5,5,5)
}
rateDistortionEmpiricalPlot <- function(){
  theoretical <- log2(1000/seq(0.001,1000,length.out = 1000))
  e_points <- c(15,12,10,7,5,3,11,6,19,7.6,11,17,16,17,16.5,18.3,19,15.5,16,14,15)
  d_points <- c(30,110,200,400,620,900,600,890,400,750,380,170,220,230,260,280,350,330,300,380,399)
  o <- order(d_points)
  d_points <- d_points[o]
  e_points <- e_points[o]
  empirical <- sapply(30:949,function(i)min(e_points[which(d_points <= i)]))
  
  
  
  savepdf("rdempirical",12,6)
  par(mar=c(3.5,4.2,0.1,0.1))
  plot(theoretical,col='red',type='l',xlab=latex2exp::TeX("$d(X,\\hat{X}),\\; d(x,r)$"),ylab=latex2exp::TeX("$H(\\hat{X}),\\; h(c)$"),xaxt='n',yaxt='n')
  axis(1,cex.axis=0.75)
  axis(2,cex.axis=0.75)
  points(d_points,e_points,col='purple')
  lines(30:949,empirical,col='blue',lty=3)
  legend("topright",c("RD function (theoretical)","RD function (empirical)","All possible clusterings"),lty=c(1,3,NA),pch=c(NA,NA,1),
         col=c("red","blue","purple"),cex = 1)
  dev.off()
  
  
}
rateDistortionLhullPlot <- function(){
  theoretical <- log2(1000/seq(0.001,1000,length.out = 1000))
  e_points <- c(15,12,10,7,5,3,11,6,19,7.6,11,17,16,17,16.5,18.3,19,15.5,16,14,15)
  d_points <- c(30,110,200,400,620,900,600,890,400,750,380,170,220,230,260,280,350,330,300,380,399)
  o <- order(d_points)
  d_points <- d_points[o]
  e_points <- e_points[o]
  empirical <- sapply(30:949,function(i)min(e_points[which(d_points <= i)]))
  
  
  
  savepdf("rdcurve",12,6)
  par(mar=c(3.5,4.2,0.1,0.1))
  plot(theoretical,col='red',type='l',xlab=latex2exp::TeX("$d(X,\\hat{X}),\\; d(x,r)$"),ylab=latex2exp::TeX("$H(\\hat{X}),\\; h(c)$"),xaxt='n',yaxt='n')
  axis(1,cex.axis=0.75)
  axis(2,cex.axis=0.75)
  points(d_points,e_points,col='purple')
  points(c(lhull(d_points,e_points)$curve[,1],d_points[c(8,19,5)]),c(lhull(d_points,e_points)$curve[,2],e_points[c(8,19,5)]),col='brown',pch=4)
  lines(30:949,empirical,col='blue',lty=3)
  points(lhull(d_points,e_points)$curve,type='l',col='darkgreen',lty=5)
  legend("topright",c("RD function (theoretical)","RD function (empirical)","All possible clusterings","Computed Clusterings","RD hull"),lty=c(1,3,NA,NA,5),
         pch=c(NA,NA,1,4,NA),col=c("red","blue","purple","brown","darkgreen"),cex = 1)
  dev.off()
  
  
}

rhoPlot <- function(){
  require(shape)
  require(pBrackets)
  require(latex2exp)
  theoretical <- log2(1000/seq(0.001,1000,length.out = 1000))
  e_points <- c(15,12,10,7,5,3,11,6,19,7.6,11,17,16,17,16.5,18.3,19,15.5,16,14,15)
  d_points <- c(30,110,200,400,620,900,600,890,400,750,380,170,220,230,260,280,350,330,300,380,399)
  o <- order(d_points)
  d_points <- d_points[o]
  e_points <- e_points[o]
  #empirical <- sapply(30:949,function(i)min(e_points[which(d_points <= i)]))
  
  
  
  savepdf("rho",24,12)
  par(mar=c(3.5,4.2,0.1,0.1))
  plot(theoretical,col='red',type='l',xlab=latex2exp::TeX("$d(X,\\hat{X}),\\; d(x,r)$"),ylab=latex2exp::TeX("$H(\\hat{X}),\\; h(c)$"),xaxt='n',yaxt='n')
  axis(1,cex.axis=0.95)
  axis(2,cex.axis=0.95)
  points(d_points,e_points,col='purple')
  #lines(30:949,empirical,col='blue',lty=3)
  points(lhull(d_points,e_points)$curve,type='l',col='darkgreen',lty=5)
  Arrows(400,7,400,9,col=rgb(0.8,0.4,0,1),lwd=2.5,arr.length = 0.3)
  Arrows(400,9.45,245,9.45,col=rgb(0,0.4,0.8,1),lwd=2.5,arr.length = 0.3)
  brackets(410,9.15,410,7.05,h=15,type = 1,lwd=2,col=rgb(0,0,0,0.25))
  brackets(240,9.65,395,9.65,h=1.5,type = 1,lwd=2,col=rgb(0,0,0,0.25))
  text(515,8.2,TeX("$h(c'_{(i,j)}) - h(c_i)$"),cex=1.5)
  text(310,12,TeX("$-d(x_j,r_{c_{i,j}})$"),cex=1.5)
  
  thecolor <- rgb(0.3,0.1,0.05)
  
  Arrows(520,6.5,420,6.9,lwd=3,arr.length = 0.2,col=thecolor)
  text(530,6.0,TeX("This is clustering $(c_i,r_i)$"),cex=1.5,col=thecolor,pos=4)
  
  Arrows(700,10,605,8.5,lwd=3,arr.length = 0.2,col=thecolor)
  text(820,10.5,"This is how the entropy changes",cex=1.5,col=thecolor)
  
  Arrows(490,15,370,12.3,lwd=3,arr.length = 0.2,col=thecolor)
  text(495,16,"If the distortion changes by this much",cex=1.5,pos = 4,col=thecolor)
  text(495,14.5,TeX("then $x_j$ is an outlier"),cex=1.5,pos=4,col=thecolor)
  
  idx <- 30:900
  h <- lhull(d_points,e_points)$curve
  #print(h)
  polygon(c(idx,rev(idx)),y=c(theoretical[idx],rev(approx(h[,1],h[,2],n=length(idx))$y)), col = rgb(1.0,0.8,0.0,0.4),border = rgb(0.1,0.1,0.1,0.2))
  #Arrows(200,7,225,9,lwd=3,arr.length = 0.2,col=thecolor)
  text(200,3.5,TeX("In this area $\\hat{\\rho}(x,\\cdot,\\cdot,\\underline{c},\\underline{r}) \\geq 1$ holds"),cex=1.5,pos=4,col=rgb(0.2,0.2,0.2))
  
  legend("topright",c("RD function (theoretical)","All possible clusterings","RD hull"),lty=c(1,NA,5),
         pch=c(NA,1,NA),col=c("red","purple","darkgreen"),cex = 1)
  dev.off()
}

dbPlot <- function(){
  savepdf("dbPlot",12,6)
  par(mfrow=c(1,2),mar=c(1.2,0.1,0.1,0.1))
  cmp1 <- dbscanCompression(coco,2,3)
  cmp2 <- cmp1
  cmp1$representation <- meanCenters(coco,cmp1$cluster)
  clusterPurgingParametric(coco,k=2.2,clustering = cmp1,pl=TRUE,add=TRUE,xaxt='n',yaxt='n')
  title(xlab=expression(paste("Centroid Representation")), line=0.5, cex.lab=1.1)
  clusterPurgingParametric(coco,k=2.2,clustering = cmp2,pl=TRUE,add=TRUE,xaxt='n',yaxt='n')
  title(xlab=expression(paste("1-NN Representation")), line=0.5, cex.lab=1.1)
  
  dev.off()
}
compareDBCPPlot <- function(){
  require(shape)
  require(pBrackets)
  require(latex2exp)
  
  #dev.new()
  #par(mfrow=c(1,2))
  savepdf("compareDBCP",24,12)
  data <- as.matrix(read.table('data.csv',sep=','))
  par(mar=c(1.2,1.3,0.1,0.1))
  cmp <- dbscanCompression(data,20,0.8)
  plot(data,asp=1,col=c("black","orange","magenta")[cmp$cluster])
  clusterPurging(data,clusterings = list(cmp,dbscanCompression(data,-1,0.8,perturbationMaximal(data,cmp))),pl=TRUE,add=TRUE,onlyZone=TRUE)
  
  o1 <- cmp$cluster %in% which(table(cmp$cluster)==1)
  L <- list()
  L[[1]] <- list()
  L[[1]]$centers <- data[which(!(o1)),]
  L[[1]]$radii <- rep(0.8,nrow(L[[1]]$centers))
  L[[1]]$outliers <- which(o1)
  
  gold <- rgb(0.95,0.55,0.05)
  createAreaPlotMult(data,L,integer(0),add = TRUE,cols='cyan',onlyZone = TRUE)
  leg <- legend(x = "topright",legend = c("Clusters",TeX("$\\epsilon$ Region"),"Purging Boundary","Outliers (both)", "Outliers (only CP)","Perturbed Obs."),
                pch=c(1,NA,NA,4,4,8),lty = c(NA,1,1,NA,NA,NA),lwd=c(1,1,1,4,4,1),col=c("black","cyan","darkgreen","red","red",gold),cex = 1.25)
  #legend(x = "topright",legend = c("data","eps Region","Purging Boundary","Outliers (both)", "Outliers (only CP)"),
  #       pch=c(1,NA,NA,4,4),lty = c(NA,1,1,NA,NA),lwd=c(1,1,1,1,4),col=c("black","cyan","green","cyan","red"))
  
  points(leg$text$x[1]-0.85,leg$text$y[1],col='orange',pch=1, cex=1.25)
  points(leg$text$x[1]-1.25,leg$text$y[1],col='magenta',pch=1,cex=1.25)
  points(leg$text$x[4]-1.05,leg$text$y[4],col='cyan',pch=4,cex=1.25,lwd=2)
  
  thecolor <- rgb(0.3,0.1,0.05)
  
  text("These clusters where" ,x=10,y=2,cex=1.5,pos = 4,col=thecolor)
  text("found by DBSCAN." ,x=10,y=1.25,cex=1.5,pos = 4,col=thecolor)
  Arrows(9.9,1.9,7,2.2,lwd=3,arr.length = 0.2,col="orange")
  #Arrows(9.9,1.9,7,0.8,lwd=3,arr.length = 0.2,col="black")
  Arrows(9.9,1.9,7,0,lwd=3,arr.length = 0.2,col="magenta")
  diagram::curvedarrow(c(9.9,1.9),c(3,-3),arr.pos = 1.0,curve = -0.2,arr.type='curved',arr.length=0.2)
  
  text("Everything outside of" ,x=-13,y=1,cex=1.5,pos = 4,col="darkgreen")
  text("this region is not" ,x=-13,y=0.25,cex=1.5,pos = 4,col="darkgreen")
  text("well represented." ,x=-13,y=-0.50,cex=1.5,pos = 4,col="darkgreen")
  Arrows(-6,2,-4.65,2.9,lwd=3,arr.length = 0.2,col="darkgreen")
  
  text("This outlier was found by DBSCAN and CP." ,x=-13,y=9,cex=1,pos = 4,col=thecolor)
  Arrows(-6.5,8.5,-5,8.2,lwd=3,arr.length = 0.2,col=thecolor)
  
  text("This outlier was found only by CP." ,x=-13,y=-4,cex=1,pos = 4,col="red")
  text("It was also used for pertubation." ,x=-13,y=-4.5,cex=1,pos = 4,col=gold)
  Arrows(-6.7,-3.5,-4.1,-0.7,lwd=3,arr.length = 0.2,col="red")
  
  points(data[1707,1],data[1707,2],col=gold,pch=8,lwd=1)
  
  dev.off()
}

interpretationPlot <- function(){
  set.seed(0815)
  cluster1 <- sweep(circleNoise(300)*2.4,MARGIN = 2,c(18,13.5),FUN = "+")
  cluster2 <- sweep(circleNoise(30)*2.5,MARGIN = 2,c(2,1.5),FUN = "+")
  o1 <- c(10,10)
  o2 <- c(5,5)
  o3 <- c(-5,0)
  
  data <- rbind(cluster1,cluster2,o1,o2,o3)
  clustering <- dbscanCompression(data,3,5.5)
  perturbation <- dbscanCompression(data,3,5.5,cluster = perturbationMaxMin(data,clustering))

  plot(data,asp=0,pch=1,xlim=c(-15,25),ylim=c(-15,25),col=clustering$cluster)
  #points(clustering1$centers,col='green',pch=2,lwd=2)
  #points(clustering2$centers,col='blue',pch=2,lwd=2)
  #clusterPurgingParametric(data,k=1,clustering=clustering,pl = TRUE,onlyZone=TRUE,add = TRUE)
  clusterPurging(data,list(clustering,perturbation),pl=TRUE,xlim=c(0,8),add=TRUE,onlyZone=TRUE)
}
perturbationComparisonPlot1 <- function(noise=NULL){
  source('utils.R')
  source('clusterPurging.R')
  europe <- as.matrix(read.table('europe/europe.txt',sep=' ')[1:2])
  if (is.null(noise)){
    n1 <- c(-20,40)
    n2 <- c(-10,80)
    n3 <- c( 60,41)
    n4 <- c( 70,60)
    noise <- rbind(n1,n2,n3,n4)
  }
  
  europe <- rbind(europe,noise)
  
  set.seed(0815+1)
  
  cmp1 <- kmeansCompression(europe,225,nstart = 1)
  cmp1.1 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMaxMin(europe,cmp1))
  cmp1.2 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMaxMax(europe,cmp1))
  cmp1.3 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMinMin(europe,cmp1))
  cmp1.4 <- kmeansCompression(europe,-1,nstart = 0,cluster = perturbationMinMax(europe,cmp1))
  
  savepdf("europeKM",12,6)
  par(mfrow=c(1,2),mar=c(1.2,0.1,0.1,0.1))
  
  plot(europe,asp=1,col="gray",pch='.',xaxt='n',yaxt='n',xlab='',ylab='')
  points(noise,col='red',pch=4)
  clusterPurging(europe,list(cmp1,cmp1.3),pl=TRUE,onlyZone=TRUE,add=TRUE,noOutliers=TRUE,cols="darkblue")
  title(xlab=expression(paste("min-min perturbation")), line=0.5, cex.lab=1.1)
  
  plot(europe,asp=1,col="gray",pch='.',xaxt='n',yaxt='n',xlab='',ylab='')
  points(noise,col='red',pch=4)
  title(xlab=expression(paste("max-max perturbation")), line=0.5, cex.lab=1.1)
  clusterPurging(europe,list(cmp1,cmp1.2),pl=TRUE,onlyZone=TRUE,add=TRUE,noOutliers=TRUE,cols="darkblue")
  
  dev.off()
  
  savepdf("europeKM2",12,6)
  par(mfrow=c(1,2),mar=c(1.2,0.1,0.1,0.1))
  
  plot(europe,asp=1,col="gray",pch='.',xaxt='n',yaxt='n',xlab='',ylab='')
  points(noise,col='red',pch=4)
  clusterPurging(europe,list(cmp1,cmp1.1),pl=TRUE,onlyZone=TRUE,add=TRUE,noOutliers=TRUE,cols="darkblue")
  title(xlab=expression(paste("max-min perturbation")), line=0.5, cex.lab=1.1)
  
  plot(europe,asp=1,col="gray",pch='.',xaxt='n',yaxt='n',xlab='',ylab='')
  points(noise,col='red',pch=4)
  title(xlab=expression(paste("min-max perturbation")), line=0.5, cex.lab=1.1)
  clusterPurging(europe,list(cmp1,cmp1.4),pl=TRUE,onlyZone=TRUE,add=TRUE,noOutliers=TRUE,cols="darkblue")
  
  dev.off()
}

perturbationComparisonPlot2 <- function(){
  require(shape)
  savepdf("europeSlope",12,6)
  base <- c(124069.71,7.663334)
  perturb1 <- c(124069.68,7.66410)
  perturb2 <- c(124065.21,7.66410)
  
  k1 <- 0.002754849#(perturb1[2]-base[2]) / (perturb1[1]-base[1])
  k2 <- -1.680601e-05#(perturb2[2]-base[2]) / (perturb2[1]-base[1])
  
  thecolor <- rgb(0.3,0.1,0.05)
  plot(rbind(base,perturb1,perturb2),pch=1,xlab='Distortion',ylab='Entropy')
  lines(rbind(base,perturb1),lty=2)
  lines(rbind(base,perturb2),lty=3)
  
  text("k=-0.00017",x=124066,y=7.6637,pos=4,cex=1.0,col=thecolor)
  text("k=-0.00276",x=124068.5,y=7.66376,pos=4,cex=1.0,col=thecolor)
  
  text("max-max perturbation",x=124065.9,y=7.66405,cex=0.8,col=thecolor,pos=4)
  text("min-min perturbation",x=124067.7,y=7.66405,cex=0.8,col=thecolor,pos=4)
  text("original",x=124069.0,y=7.6636,cex=0.8,col=thecolor)
  
  Arrows(124065.9,7.664065,124065.5,7.664090,lwd=1,arr.length = 0.2,col=thecolor)
  Arrows(124069.1,7.664065,124069.5,7.664093,lwd=1,arr.length = 0.2,col=thecolor)
  Arrows(124069.0,7.66355,124069.5,7.66341,lwd=1,arr.length = 0.2,col=thecolor)
  Arrows(124067,7.6637,124067.2,7.66372,lwd=1,arr.length = 0.2,col=thecolor)
  Arrows(124069.1,7.66381,124069.5,7.663830,lwd=1,arr.length = 0.2,col=thecolor)
  
  
  dev.off()
}