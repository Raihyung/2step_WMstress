rm(list=ls())

library(rstan)

dat <- read.table("DM_otto.txt", header=T, sep="\t")

# Individual Subjects
subjList <- unique(dat[,"subjID"]) # list of subjects x blocks
numSubjs <- length(subjList)           # number of subjects

# data 
Tsubj <- as.vector( rep( 0, numSubjs ) ) # number of trials for each subject

for ( sIdx in 1:numSubjs )  {
  curSubj     <- subjList[ sIdx ]
  Tsubj[sIdx] <- sum( dat$subjID == curSubj )  # Tsubj[N]
}

maxTrials <- max(Tsubj)

# for multiple subjects
level1_choice    <- array(1, c(numSubjs, maxTrials) )
level2_choice    <- array(1, c(numSubjs, maxTrials) )
state    <- array(1, c(numSubjs, maxTrials) )
reward    <- array(0, c(numSubjs, maxTrials) )

for (i in 1:numSubjs) {
  curSubj      <- subjList[i]
  useTrials    <- Tsubj[i]
  tmp          <- subset(dat, dat$subjID == curSubj)
  level1_choice[i, 1:useTrials]    <- tmp[1:useTrials, "level1_choice"]
  level2_choice[i, 1:useTrials]    <- tmp[1:useTrials, "level2_choice"]
  reward[i, 1:useTrials]    <- tmp[1:useTrials, "reward"]
  state[i, 1:useTrials]    <- tmp[1:useTrials, "state"]
}

dataList <- list(
  NS       = numSubjs,
  MT       = maxTrials,
  NT   = Tsubj,
  c1    = level1_choice,
  c2    = level2_choice,
  r    = reward,
  st = state
)


# fit

fit = stan("2stepMarkov.stan", data = dataList, iter = 2000, warmup=1000, chains=4, cores=6, thin=1, init="random")

parVals <- rstan::extract(fit, permuted=T)

alpha     <- parVals$alpha
lambda  <- parVals$lambda
betamb     <- parVals$betamb
betamf  <- parVals$betamf
beta2     <- parVals$beta2
pi <- parVals$pi

allIndPars <- array(NA, c(numSubjs, 6))
allIndPars <- as.data.frame(allIndPars)

for (i in 1:numSubjs) {
  allIndPars[i, ] <- c( mean(alpha[, i]),
                        mean(lambda[, i]),
                        mean(betamb[, i]),
                        mean(betamf[, i]),
                        mean(beta2[, i]),
                        mean(pi[, i]))
}

allIndPars           <- cbind(allIndPars, subjList)
colnames(allIndPars) <- c("alpha",
                          "lambda",
                          "betamb",
                          "betamf",
                          "beta2",
                          "pi",
                          "subjID")



## tracePlot
library(ggplot2)
alpha_mu <- traceplot(fit, pars = c("alpha_mu")) + labs(x = "", y = "") + ggtitle("alpha_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))
lambda_mu <- traceplot(fit, pars = c("lambda_mu")) + labs(x = "", y = "") + ggtitle("lambda_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))
betamb_mu <- traceplot(fit, pars = c("betamb_mu")) + labs(x = "", y = "") + ggtitle("betamb_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))
betamf_mu <- traceplot(fit, pars = c("betamf_mu")) + labs(x = "", y = "") + ggtitle("betamf_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))
beta2_mu <- traceplot(fit, pars = c("beta2_mu")) + labs(x = "", y = "") + ggtitle("beta2_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))
pi_mu <- traceplot(fit, pars = c("pi_mu")) + labs(x = "", y = "") + ggtitle("pi_mu")  + theme(plot.title = element_text(hjust=0.5, size=25))


library("gridExtra")
grid.arrange(alpha_mu,lambda_mu,betamb_mu,betamf_mu,beta2_mu,pi_mu)

## Posterior
library(bayesboot)
par(mfrow=c(3,2))
plotPost(parVals$alpha_mu, credMass=0.95, xlab="alpha_mu")
plotPost(parVals$lambda_mu, credMass=0.95, xlab="lambda_mu")
plotPost(parVals$betamb_mu, credMass=0.95, xlab="betamb_mu")
plotPost(parVals$betamf_mu, credMass=0.95, xlab="betamf_mu")
plotPost(parVals$beta2_mu, credMass=0.95, xlab="beta2_mu")
plotPost(parVals$pi_mu, credMass=0.95, xlab="pi_mu")

stan_hist(fit, "alpha")
stan_hist(fit, "lambda")
stan_hist(fit, "betamb")
stan_hist(fit, "betamf")
stan_hist(fit, "beta2")
stan_hist(fit, "pi")




