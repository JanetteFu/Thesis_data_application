## We construct the model in the Stan format  
## A Stan model contains the following sections: 1. input data (and a "transformed data" section if we need to specify some hyperparameter values in the process of priors construction); 
##                                               2. parameters (the parameters we want to estimate, and an additional "transformed parameters" section if needed);
##                                               3. model (we specify the prior distribution and the outcome distribution);
##                                               4. generated quantities (if need to generate the prediction of outcome)

Regularized_fully<-"data {
//start with the data type (int[N] for 1-D integer vector with length of N, matrix for 2-D matrix, the <lower,upper> specifies the accepatable range of input)
  
  int<lower = 1> N; // Number of observations
  int<lower = 0> X_mis[N]; //missing conditions of observation: 0: complete, 1:sex_active missing, 2:current_partner missing, 3:gender missing
  int<lower=1> Ms; // Number of complete features
  int<lower=1> M; // Number of features in total
  matrix[N, Ms] X; // matrix of features for observed outcomes
  int<lower=1> N_a; // column position of sex_active (1st missing variable)
  int<lower=1> N_p; // column position of current_partner (2nd missing variable)
  int<lower=1> N_g; // column position of gender (3rd missing variable)
  int<lower = 0> active[N]; //vector of sex_active, since Stan does not accept NA values, we assign all NA values to 0 as default
  int<lower = 0> gender[N]; //vector of gender
  int<lower = 0> partner[N]; //vector of current_partner
  int<lower =0, upper = 1> y_obs[N]; //vector of outcome
}

transformed data {
// Here we list some hyperparameters used in constructing a regularized Horseshoe prior, with detailed explanation in Chapter 3.2.2
  real tau0 = 0.001;      
  real slab_scale = 4;    
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  // parameters used in the middle process of regularized Horseshoe prior
  
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real<lower=0> tau_tilde;
  
  real alpha; //the intercept in the logistic model
  real<lower=0, upper=1> p_g; //probablity of gender = 1
  real<lower=0, upper=1> p_a; //probability of sex_active = 1
  real<lower=0, upper=1> p_p; //probability of current_partner = 1
}

transformed parameters {
  // estimating the preditors' coefficients, which is assigned with a regularized Horseshoe prior
  
  vector[M] beta;
  vector[M] lambda_tilde;

{
    real tau = tau0 * tau_tilde;

    real c2 = slab_scale2 * c2_tilde;

    lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

    beta = tau * lambda_tilde .* beta_tilde;
}
}
model {
  //distribution of each parameter.
  
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 1);
  
  p_g ~ beta(1,1);
  p_a ~ beta(1,1);
  p_p ~ beta(1,1);
  
  // the outcome model, fit for each observation 
  for (n in 1:N) {
  // if observation n contains missing value, we have missing value follows a bernoulli distribution. We treat it as a mixture model:
  // p(y|X_obs,X_mis) = product of (Pr(X_mis = 1)*Logit(y | X_obs, 1) + Pr(X_mis = 0)*Logit(y | X_obs, 0)  
  // Note: if the outcome is continuous, we change the link function the logit regression to y ~ N(f(X_obs,X_mis), variance)  
  // This strategy apply to all missing values  
        if ( X_mis[n]==1 ) {
            // if gender missing
            target += log_mix( p_g ,
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + beta[N_g] + active[n]*beta[N_a] + partner[n]*beta[N_p] + alpha),
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + active[n]*beta[N_a] + partner[n]*beta[N_p] + alpha));
  }
        if ( X_mis[n]==2 ) {
            // if active missing
            target += log_mix( p_a ,
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + beta[N_a] + gender[n]*beta[N_g] + partner[n]*beta[N_p] + alpha),
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + gender[n]*beta[N_g] + partner[n]*beta[N_p] + alpha));
        }
        if ( X_mis[n]==3 ) {
            // if partner missing
            target += log_mix( p_p ,
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + beta[N_p] + gender[n]*beta[N_g] + active[n]*beta[N_a] + alpha),
                    bernoulli_logit_lpmf( y_obs[n] | X[n,] * beta[1:Ms] + gender[n]*beta[N_g] + active[n]*beta[N_a] + alpha));
        }
      
      // if observation n does not contain missing value, fit as usual    
      if(X_mis[n]==0) {
            // x_i complete
            gender[n] ~ bernoulli(p_g);
            active[n] ~ bernoulli(p_a);
            partner[n] ~ bernoulli(p_p);
             
  y_obs[n] ~ bernoulli_logit(X[n,] * beta[1:Ms]  + gender[n]*beta[N_g] + active[n]*beta[N_a] + partner[n]*beta[N_p] + alpha);
  }
  }
}
  "

### model implementation process

library(rstan)

## Now we implement the model with an artificial data 
## 1. source the stan model 
Imputation_model <- stan_model(model_code = Regularized_fully)
## 2. (1) generate artificial data: 
X <- matrix(nrow=100,ncol=20)
for(j in 1:17){
  temp <- sample(c(1,2,3),1);
  if(temp == 1){
    X[,j] = rnorm(100);
  }
  if(temp == 2){
    X[,j] = rnorm(100,mean=0,sd=3);
  }
  if(temp == 3){
    X[,j] = rbinom(100,1,0.25);
  }
}
for(j in 18:20){
  X[,j] = rbinom(100,1,0.25);
}

beta = c(runif(2,min=-2,max=(-0.8)), runif(2,max=1.8,min=1), rnorm(14,mean=0,sd=0.1), 
         1,-1)
pr = apply(X*beta,1,sum)
##    (2) decide some observations missing  
X_com = X[,1:17] 
# let around 7 percent of each variable missing
mis_condition = sample(c(rep(0,12),1,2,3),100,replace = TRUE)
active = X[,18]
gender = X[,19]
partner = X[,20]

y = rbinom(100,1,exp(X%*%beta)/(1+exp(X%*%beta)))

##    (2) enlist input data 
data_imputation = list(N=100,X_mis=mis_condition,Ms=17,M=20,X=X_com,N_a=18,N_g=19,N_p=20,active=active,gender=gender,partner=partner,y_obs=y)

## 3. fit the stan model
fit_imputation <- sampling(
  Imputation_model
  , data = data_imputation
  , iter = 3000
  , cores = 4
  , chains = 4
  , verbose = F,control = list(adapt_delta = 0.95))

## 4. save as RDS file 
saveRDS(fit_imputation,"fit_imputation.rds")
