## We construct the model in the Stan format  
## A Stan model contains the following sections: 1. input data (and a "transformed data" section if we need to specify some hyperparameter values in the process of priors construction); 
##                                               2. parameters (the parameters we want to estimate, and an additional "transformed parameters" section if needed);
##                                               3. model (we specify the prior distribution and the outcome distribution);
##                                               4. generated quantities (generate the prediction of outcome)

AS_spatial <- "data {
  //start with the data type (int[N] for 1-D integer vector with length of N, matrix for 2-D matrix, the <lower,upper> specifies the accepatable range of input)
  
  int<lower=1> N; // Number of observations in the data
  int<lower=1> M; // Number of features 
  matrix[N, M] X; //the matrix of parameters values
  int<lower =0, upper = 1> y[N]; // outcome vector
  int<lower=1> N_area;  //the number of regions in Quebec
  int<lower=0> N_edges;    // number of edges between regions: if region A, B attached
  int<lower=1, upper=N> node1[N_edges];   // the adjacency indicator between node1[i] (region) and node2[i] (region)
  int<lower=1, upper=N> node2[N_edges];   // and node1[i] < node2[i]
  int S[N];   //the region patient i belongs to
}

transformed data {
  // Here we list some hyperparameters used in constructing a regularized Horseshoe prior, with detailed explanation in Chapter 3.2.2
  real tau0 = 0.001;     
  real slab_scale = 4;  
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;     
  real half_slab_df = 0.5 * slab_df;
}

parameters {
 // parameters used in the middle process of regularized Horseshoe prior
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  
  real alpha;   //the intercept in the logistic model  
  
  real<lower=0> sigma;      // the global spatial effect
  vector[N_area] zeta;      // region-specific spatial effects, with sigma*zeta equals to regional random effects
}

transformed parameters {
  // estimating the preditors' coefficients, which is assigned with a regularized Horseshoe prior 
  vector[M] beta;
  real tau = tau0 * tau_tilde;
  
  real c2 = slab_scale2 * c2_tilde;
  
  vector[M] lambda_tilde;
  lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) *
                                                 square(lambda)) );
  
  beta = tau * lambda_tilde .* beta_tilde;
}

model {
  //distribution of each parameter. 
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  sigma ~ gamma(1,1);
  
  //the distribution of regional random effects 
  target += -0.5 * dot_self(zeta[node1] - zeta[node2]);
  sum(zeta) ~ normal(0, 0.001 * N);  
  
  // the outcome model, fit for each observation  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha + sigma*zeta[S[n]]);
  } 
}

generated quantities {
 // generates pi_i for y_i ~ Bernoulli(p_i)
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + sigma*zeta[S[n]]);
  }
}"

### model implementation process

library(rstan)

## Now we implement the model with an artificial data 
## 1. source the stan model 
spatial_model <- stan_model(model_code = AS_spatial)
## 2. (1) generate artificial data: 

## first generate the predictors: normally distributed continuous predictors (like age), and bernoulli distributed predictors (like sex)
X <- matrix(nrow=100,ncol=20)
for(j in 1:20){
  temp <- sample(c(1,2,3),1);
  if(temp == 1){
    X[,j] = rnorm(100);
  }
  if(temp == 2){
    X[,j] = rnorm(100,mean=0,sd=3);
  }
  if(temp == 3){
    X[,j] = rbinom(100,1,0.5);
  }
}

beta = c(runif(3,min=-1.2,max=(-0.3)), runif(3,max=1.5,min=0.2), rnorm(14,mean=0,sd=0.1))
pr = apply(X*beta,1,sum)

# Then generate spatial effects, we assmue 5 regions, with region 1 motropolitan, and region 5 remote areas 
# we design the following adjacency strcuture: region 1 adjacent to 2 4 , 2 adjacent to 1 3 4 5 , 3 to 2 5 , 4 to 1 2 5 and region 5 adjacent to 3 4  

Neigh <- c(2, 4, 1, 3, 4, 5, 2, 5, 1, 2, 5, 3, 4)
Num <- c(2,4,2,3,2)

nbs <- mungeCARdata4stan(Neigh, Num) #generate nodes details, the function is listed at the end, credit to Laís Picinini Freitas

## randomly assign each patient a region
area <- round(runif(100, min = 0.5, max = 5.5))
spatial_effect <- c()
for(i in 1:100){
  if(area[i]==1){
    spatial_effect[i] = 1
  }
  if(area[i]==2){
    spatial_effect[i] = 0.5
  }
  if(area[i]==3){
    spatial_effect[i] = -0.5
  }
  if(area[i]==4){
    spatial_effect[i] = 0
  }
  if(area[i]==5){
    spatial_effect[i] = -1
  }
}

y = rbinom(100,1,exp(pr+spatial_effect)/(1+exp(pr+spatial_effect)))
##    (2) enlist input data 
data_spatial = list(N=100,M=20,X=X,y=y,N_area=5,N_edges=7,node1=nbs$node1,node2=nbs$node2,S=area)

## 3. fit the stan model
fit_spatial <- sampling(
  spatial_model
  , data = data_spatial
  , iter = 3000
  , cores = 3
  , chains = 4
  , verbose = F,control = list(adapt_delta = 0.95))

## 4. save as RDS file 
saveRDS(fit_spatial,"fit_spatial.rds")


## the node generating function, credit to Laís Picinini Freitas, used in paper "Spatio-temporal modelling of the first Chikungunya epidemic in 
## an intra-urban setting: The role of socioeconomic status, environment and temperature", published on June 18, 2021 in journal "PLOS Neglected Tropical Diseases"

#adjBUGS: adjacent neighbors, numBUGS: number of neighbors
mungeCARdata4stan = function(adjBUGS,numBUGS) {
  N = length(numBUGS);
  nn = numBUGS;
  N_edges = length(adjBUGS) / 2;
  node1 = vector(mode="numeric", length=N_edges);
  node2 = vector(mode="numeric", length=N_edges);
  iAdj = 0;
  iEdge = 0;
  for (i in 1:N) {
    for (j in 1:nn[i]) {
      iAdj = iAdj + 1;
      if (i < adjBUGS[iAdj]) {
        iEdge = iEdge + 1;
        node1[iEdge] = i;
        node2[iEdge] = adjBUGS[iAdj];
      }
    }
  }
  return (list("N"=N,"N_edges"=N_edges,"node1"=node1,"node2"=node2));
}
