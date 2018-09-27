// Credit: adapted from Otto et al., 2013. 

data {
	int NS;      // subjects
	int MT;      // max trial
	
	int NT[NS];   // trial per subject

	int r[NS,MT];    // reward
	int c1[NS,MT];    // level1 choice
	int c2[NS,MT];    // level2 choice
	int st[NS,MT];   // state 
}

parameters {
	 
// Hyperparameters
  real<lower=0,upper=1> alpha_mu;       
	real<lower=0> alpha_s;             
	real<lower=0,upper=1> lambda_mu;            
	real<lower=0> lambda_s;                  

	real betamb_mu;                          
	real<lower=0> betamb_s;                 
	real betamf_mu;                          
	real<lower=0> betamf_s;                      
	real beta2_mu;                          
	real<lower=0> beta2_s;                 

	real pi_mu;                         
	real<lower=0> pi_s;               
	

// Individual parameters
	real<lower=0,upper=1> alpha[NS];     
	real<lower=0,upper=1> lambda[NS];        

	real betamb[NS];                      
	real betamf[NS];                      
	real beta2[NS];                      

	real pi[NS];                      
}

transformed parameters {
	real alpha_A;                      
	real alpha_B;                      
 	real lambda_A;                      
 	real lambda_B;                      

	alpha_A = alpha_mu * pow(alpha_s,-2);
	alpha_B = pow(alpha_s,-2) - alpha_mu;

	lambda_A = lambda_mu * pow(lambda_s,-2);
	lambda_B = pow(lambda_s,-2) - lambda_mu;
}

model {
  betamb_mu ~ normal(0,100);           
	betamb_s ~ cauchy(0,2.5);            
	
	betamf_mu ~ normal(0,100);         
	betamf_s ~ cauchy(0,2.5);            
	
  beta2_mu ~ normal(0,100);           
	beta2_s ~ cauchy(0,2.5);           
	
	pi_mu ~ normal(0,100);           
	pi_s ~ cauchy(0,2.5);          
	

	for (s in 1:NS) {

		int pc;             
		int tcounts[2,2];   // transition counts
		real v_mb[2];       // model-based values
		real v_mf1[2];      // 1st-stage model-free values
		real v_mf2[2,2];    // 2nd-stage model-free values

		alpha[s] ~ beta(alpha_A,alpha_B);
		lambda[s] ~ beta(lambda_A,lambda_B);

		betamb[s] ~ normal(betamb_mu,betamb_s);
		betamf[s] ~ normal(betamf_mu,betamf_s);
		beta2[s] ~ normal(beta2_mu,beta2_s);

		pi[s] ~ normal(pi_mu,pi_s);

		for (i in 1:2) for (j in 1:2) tcounts[i,j] = 0;
		for (i in 1:2) {v_mb[i] = 0; v_mf1[i] = 0;}
		for (i in 1:2) for (j in 1:2) v_mf2[i,j] = 0;
		
		pc = 0;

		for (t in 1:NT[s]) {
		        
		        if(tcounts[1,1]+tcounts[2,2]-tcounts[1,2]-tcounts[2,1] > 0) {
		          v_mb[1] = fmax(v_mf2[1,1],v_mf2[1,2]);
		          v_mb[2] = fmax(v_mf2[2,1],v_mf2[2,2]);
		        }
		        else {
		          v_mb[1] = fmax(v_mf2[2,1],v_mf2[2,2]);
		          v_mb[2] = fmax(v_mf2[1,1],v_mf2[1,2]);
		        } 

			c1[s,t] ~ bernoulli_logit((betamb[s] ) * (v_mb[2] - v_mb[1]) 
						+ (betamf[s] ) * (v_mf1[2] - v_mf1[1])  
						+ pi[s] * pc );

			pc = 2 * c1[s,t] - 1;

			tcounts[c1[s,t]+1,st[s,t]] = tcounts[c1[s,t]+1,st[s,t]] + 1;

			
      v_mf1[c1[s,t]+1] = v_mf1[c1[s,t]+1] * (1 - alpha[s]) + v_mf2[st[s,t],c2[s,t]+1];

			c2[s,t] ~ bernoulli_logit((beta2[s]) * (v_mf2[st[s,t],2] - v_mf2[st[s,t],1]));

      v_mf1[c1[s,t]+1] = v_mf1[c1[s,t]+1] + lambda[s] * (r[s,t] - v_mf2[st[s,t],c2[s,t]+1]);

    	v_mf2[st[s,t],c2[s,t]+1] = v_mf2[st[s,t],c2[s,t]+1] * (1 - alpha[s]) + r[s,t];
    	
    	
			
		}
			
	}
}

