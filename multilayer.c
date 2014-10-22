#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ray.h"

#define MULTILAYER_HOME
#include "bi_ccd.h"
#undef MULTILAYER_HOME

// #define NOLOSSY

#include "multilayer.h"
#include "complex_n_server.h"

// file scope err string

char err[2048];
float ml_Si_T;

void compare_float_complex (char *s,float f,complex *c);

void compute_layer_internal_reflectivities(multilayer *mlstruct,int layer_ix,
					   complex rho_q[],complex rho_qp[]) {
  layer_pars *the_layer=&(mlstruct->lp[layer_ix]);
  complex *Sqn[2][2],SQN[2][2];
  complex *inv_S1q[2][2],inv_S1Q[2][2];
  complex *Lq[2][2],*S1q[2][2],*S1n[2][2];
  complex *En[2],EN[2];
  complex *E1[2],E_1[2];
  complex *Eq[2],EQ[2];
  complex *Eqp[2],EQp[2];
  complex cpx_null;
  int     i,j,s;

  cpx_null.comp[0]=cpx_null.comp[1]=0;

  for (i=0;i<2;i++) {
    En[i] = &EN[i];    E1[i] = &E_1[i];
    Eq[i] = &EQ[i];    Eqp[i] = &EQp[i];
    for (j=0;j<2;j++) {
      // connect & initialize
      Sqn[i][j]     = &SQN[i][j];
      Lq[i][j]      = &the_layer->L_mx[i][j];
      inv_S1q[i][j] = &inv_S1Q[i][j];
    }
  }

  E1[0]->comp[0]=1;  E1[0]->comp[1]=0;
  E1[1]->comp[0]=0;  E1[1]->comp[1]=0;
  En[0]->comp[0]=0;  En[0]->comp[1]=0;
  En[1]->comp[0]=1;  En[1]->comp[1]=0;
  
  for (s=0;s<2;s++) {
    for (i=0;i<2;i++) {
      for (j=0;j<2;j++) {
	// connect & initialize
	S1q[i][j] = &the_layer->S1Q[i][j][s];
	S1n[i][j] = &mlstruct->S[i][j][s];
	memcpy(Sqn[i][j],&cpx_null,sizeof(complex));
      }
    }
    // prepare the matrices
    // compute S1Q^{-1} and SQN
    cpx_2x2_mat_inv(S1q,inv_S1q);
    if (0) {
      // this here algorithm
      // runs into problems if Lq represents a lot of attenuation
      // left in here in case we want a test to be made (computing attenuation)
      // following which a decision for which calculation to perform
      cpx_2x2_mat_prod(S1q,Lq,Sqn);
      cpx_2x2_mat_inv(Sqn,Sqn);
      cpx_2x2_mat_prod(Sqn,S1n,Sqn);
    } else {
      complex *H[2][2],*L[2][2],HC[2][2];
      int k,l,m;
      for (l=0;l<2;l++) {
	for (m=0;m<2;m++) {
	  H[l][m] = &HC[l][m];
	}
      }
      for (k=layer_ix;k<mlstruct->nlayer;k++) {
	for (l=0;l<2;l++) {
	  for (m=0;m<2;m++) {
	    if (0) {
	      // float
	      H[l][m]->comp[0] = mlstruct->lp[k].iface_p[1]->re_H_mx[l][m][s];
	      H[l][m]->comp[1] = 0;
	    } else {
	      // complex
	      memcpy(H[l][m],
		     &mlstruct->lp[k].iface_p[1]->H_mx[l][m][s],
		     sizeof(complex));
	    }
	    L[l][m] = &mlstruct->lp[k].L_mx[l][m];
	  }
	}
	if (k==layer_ix) {
	  // don't include L in running matrix
	  for (l=0;l<2;l++) {
	    for (m=0;m<2;m++) {
	      memcpy(Sqn[l][m],H[l][m],sizeof(complex));
	    }
	  }
	} else {
	  // apply L to running matrix followed by H
	  cpx_2x2_mat_prod(Sqn,L,Sqn);
	  cpx_2x2_mat_prod(Sqn,H,Sqn);
	}
      }
      // Sqn is ready to use.
      //      fprintf(stderr,"polarization %d:\n",s);
      //      display_mat(Sqn);
    }
    // use the matrices
    for (i=0;i<2;i++) {
      memcpy(Eq[i],&cpx_null,sizeof(complex));
      memcpy(Eqp[i],&cpx_null,sizeof(complex));
      for (j=0;j<2;j++) {
	if (1) {
	  memcpy(Eqp[i],
		 cpx_increment(Eqp[i],cpx_product(Sqn[i][j],En[j])),
		 sizeof(complex));
	  memcpy(Eq[i],
		 cpx_increment(Eq[i],cpx_product(inv_S1q[i][j],E1[j])),
		 sizeof(complex));
	}
      }
    }
    memcpy(&rho_q[s],cpx_ratio(Eq[1],Eq[0]),sizeof(complex));
    memcpy(&rho_qp[s],cpx_ratio(Eqp[0],Eqp[1]),sizeof(complex));
  }
  // finished. reflectivities are stored in rho_q[] and rho_qp[]
}

void compute_multilayer(float theta,float lambda,
			multilayer *mlstruct) {
  media_pars *initial,*final;
  layer_pars *layer=mlstruct->lp;
  interface_pars *interface=mlstruct->ip;
  int nlayer=mlstruct->nlayer;
  complex cpx_null;
  cpx_null.comp[0]=cpx_null.comp[1]=0;

  initial=&(mlstruct->mp[0]);
  final=&(mlstruct->mp[mlstruct->nlayer+2-1]);

  int i,j;
  int reevaluate_geometry=0;
  //  int reevaluate_coefficients=0;
  media_pars *medium;

  if (lambda != mlstruct->lambda) {
    // reevaluate optical constants in all media
    for (i=0;i<nlayer+2;i++) {
      medium=&(mlstruct->mp[i]);
      if (lambda != medium->lambda) {

	if (medium->optical_constants == const_n_func) {
	  medium->n.comp[0]=medium->const_n_val; // real comp
	  medium->n.comp[1]=0;                   // imag comp
	} else {
	  (*medium->optical_constants)(lambda,&medium->n);
	}

	medium->lambda=lambda;
      }
    }
    mlstruct->lambda=lambda;
    reevaluate_geometry=1;
  }
  
  // optical constants are (re)evaluated. next do geomoetry
  if ((theta != mlstruct->theta) || (reevaluate_geometry==1)) {
    float n_sin_theta,this_theta;
    complex cpx_sin_theta,cpx_n_sin_theta,cpx_this_theta;
    // do float..
    n_sin_theta=initial->n.comp[0]*sin(theta);
    // do complex..
    cpx_set(&cpx_this_theta,theta,0.0);
    cpx_sin(&cpx_sin_theta,&cpx_this_theta);
    memcpy(&cpx_n_sin_theta,cpx_product(&initial->n,&cpx_sin_theta),sizeof(complex));
    // compare float & complex
    compare_float_complex("n sin(theta)",n_sin_theta,&cpx_n_sin_theta);
    
    for (i=0;i<nlayer+2;i++) {
      medium=&(mlstruct->mp[i]);
      // do float..
      this_theta=asin(n_sin_theta/medium->n.comp[0]);
      // do complex..
      memcpy(&cpx_sin_theta,cpx_ratio(&cpx_n_sin_theta,&medium->n),
	     sizeof(complex));
      // compare
      compare_float_complex("sin(theta)",sin(this_theta),&cpx_sin_theta);
      // float
      if (this_theta != medium->theta) {
	medium->theta=this_theta;
	// used to use this place to set flags that would trigger reevaluation
	// of coefficients rho & tau - but it wasn't general enough
	// to handle all cases of variables changing. removed those assignments.
      }
      // complex
      if (memcmp(&medium->cpx_sin_theta,&cpx_sin_theta,sizeof(complex)) != 0) {
	memcpy(&medium->cpx_sin_theta,&cpx_sin_theta,sizeof(complex));
      }
    }
    {
      float n_over_np,*a,*b;
      complex *ca,*cb;
      complex unity;
      complex n_ratio;

      cpx_set(&unity,1.0,0.0);

      for (i=0;i<nlayer+1;i++) {
	interface=&(mlstruct->ip[i]);
	// float
	n_over_np = (interface->media_p[0]->n.comp[0]/
		     interface->media_p[1]->n.comp[0]);
	// complex
	memcpy(&n_ratio,
	       cpx_ratio(&interface->media_p[0]->n,&interface->media_p[1]->n),
	       sizeof(complex));
	// compare
	compare_float_complex("n_over_np vs. n_ratio",n_over_np,&n_ratio);
	// do float
	a = &(interface->a);
	b = &(interface->b);
	*a = (cos(interface->media_p[1]->theta)/
	      cos(interface->media_p[0]->theta))/n_over_np;
	*b = interface->a*pow(n_over_np,2);
	// do complex
	complex cpx_cos_theta,cpx_cos_theta_p;
	ca = &(interface->cpx_a);
	cb = &(interface->cpx_b);

	memcpy(&cpx_cos_theta,
	       cpx_sqrt(
   cpx_increment(cpx_negate(cpx_product(&interface->media_p[0]->cpx_sin_theta,
					&interface->media_p[0]->cpx_sin_theta)),
		 &unity)),
	       sizeof(complex));

	memcpy(&cpx_cos_theta_p,
	       cpx_sqrt(
   cpx_increment(cpx_negate(cpx_product(&interface->media_p[1]->cpx_sin_theta,
					&interface->media_p[1]->cpx_sin_theta)),
		 &unity)),
	       sizeof(complex));

	// compare float & complex.
	compare_float_complex("cpx_cos_theta  ",cos(interface->media_p[0]->theta),&cpx_cos_theta);
	compare_float_complex("cpx_cos_theta_p",cos(interface->media_p[1]->theta),&cpx_cos_theta_p);

	// do float..
	// s-polarization
	interface->tau[0] = 2/(1+*a);
	interface->rho[0] = (1-*a)/(1+*a);
	// p-polarization
	interface->tau[1] = 2*(n_over_np)/(1+*b);
	interface->rho[1] = (1-*b)/(1+*b);

	// do complex..
	complex numer,denom,cpx_tmp,*cpx_tau,*cpx_rho;
	// compute a
	memcpy(&numer,
	       cpx_product(&interface->media_p[1]->n,&cpx_cos_theta_p),
	       sizeof(complex));
	memcpy(&denom,
	       cpx_product(&interface->media_p[0]->n,&cpx_cos_theta),
	       sizeof(complex));
	memcpy(ca,cpx_ratio(&numer,&denom),sizeof(complex));

	// compute b
	memcpy(&cpx_tmp,cpx_product(&n_ratio,&n_ratio),sizeof(complex));
	memcpy(cb,cpx_product(ca,&cpx_tmp),sizeof(complex));

	cpx_tau=interface->cpx_tau;
	cpx_rho=interface->cpx_rho;
	
	// for s-polarization
	// first calculate denominator for both tau & rho
	memcpy(&denom,ca,sizeof(complex));
	cpx_increment(&denom,&unity);
	// numerator for rho
	memcpy(&numer,ca,sizeof(complex));
	memcpy(&numer,cpx_negate(&numer),sizeof(complex));
	cpx_increment(&numer,&unity);
	// compute rho for s-polarization
	memcpy(&cpx_rho[0],cpx_ratio(&numer,&denom),sizeof(complex));
	// compute tau for s-polarization
	cpx_set(&numer,2.0,0.0);
	memcpy(&cpx_tau[0],cpx_ratio(&numer,&denom),sizeof(complex));
	// for p-polarization
	// first calculate denominator for both tau & rho
	memcpy(&denom,cb,sizeof(complex));
	cpx_increment(&denom,&unity);
	// numerator for rho
	memcpy(&numer,cb,sizeof(complex));
	memcpy(&numer,cpx_negate(&numer),sizeof(complex));
	cpx_increment(&numer,&unity);
	// compute rho for p-polarization
	memcpy(&cpx_rho[1],cpx_ratio(&numer,&denom),sizeof(complex));
	// compute tau for s-polarization
	cpx_set(&numer,2.0,0.0);
	memcpy(&numer,cpx_product(&numer,&n_ratio),sizeof(complex));
	memcpy(&cpx_tau[1],cpx_ratio(&numer,&denom),sizeof(complex));

	// finished computing cpx_rho and cpx_tau for both polarizations.
	// now compare.

	compare_float_complex("s-pol rho",interface->rho[0],&interface->cpx_rho[0]);
	compare_float_complex("s-pol tau",interface->tau[0],&interface->cpx_tau[0]);
	compare_float_complex("p-pol rho",interface->rho[1],&interface->cpx_rho[1]);
	compare_float_complex("p-pol tau",interface->tau[1],&interface->cpx_tau[1]);

	// what to do with the complex rhos..

	for (j=0;j<2;j++) {
	  // real
	  interface->re_H_mx[0][0][j]=interface->re_H_mx[1][1][j]=
	    1.0/interface->tau[j];
	  interface->re_H_mx[0][1][j]=interface->re_H_mx[1][0][j]=
	    interface->rho[j]/interface->tau[j];
	  // complex
	  memcpy(&interface->H_mx[0][0][j],
		 cpx_ratio(&unity,&interface->cpx_tau[j]),
		 sizeof(complex));
	  memcpy(&interface->H_mx[1][1][j],
		 &interface->H_mx[0][0][j],
		 sizeof(complex));

	  memcpy(&interface->H_mx[0][1][j],
		 cpx_ratio(&interface->cpx_rho[j],&interface->cpx_tau[j]),
		 sizeof(complex));
	  memcpy(&interface->H_mx[1][0][j],
		 &interface->H_mx[0][1][j],
		 sizeof(complex));
	  // compare:
	  compare_float_complex("H_mx[0,0]",interface->re_H_mx[0][0][j],
				&interface->H_mx[0][0][j]);
	  compare_float_complex("H_mx[0,0]",interface->re_H_mx[0][1][j],
				&interface->H_mx[0][1][j]);
	  compare_float_complex("H_mx[0,0]",interface->re_H_mx[1][0][j],
				&interface->H_mx[1][0][j]);
	  compare_float_complex("H_mx[0,0]",interface->re_H_mx[1][1][j],
				&interface->H_mx[1][1][j]);
	}
      }
      // finally evaluate the propagation matrices through each layer
      mlstruct->beta_total=0;
      for (i=0;i<nlayer;i++) {
	float ep,cp,sp;
	float tmp;
	layer=&(mlstruct->lp[i]);
	medium=layer->iface_p[0]->media_p[1]; // or, equivalently..
	medium=layer->iface_p[1]->media_p[0];
	tmp  = ((2*M_PI*layer->thickness/medium->lambda)*cos(medium->theta));
	layer->beta.comp[0]=tmp*medium->n.comp[0];
	layer->beta.comp[1]=tmp*medium->n.comp[1];

	mlstruct->beta_total+=layer->beta.comp[1];

	if (mlstruct->beta_total < 12) {
	  // originally this was used to truncate the calculation
	  // here we compute L_mx anyway, since reflectivities may
	  // be computed even behind thick layers.
	  mlstruct->eff_nlayer=i+1;
	}
	
	ep=exp(-layer->beta.comp[1]);
	cp=cos(layer->beta.comp[0]);
	sp=sin(layer->beta.comp[0]);
	
	layer->L_mx[0][0].comp[0]=+ep*cp;
	layer->L_mx[0][0].comp[1]=-ep*sp;
	layer->L_mx[1][1].comp[0]=+cp/ep;
	layer->L_mx[1][1].comp[1]=+sp/ep;
	
	memcpy(&layer->L_mx[0][1],&cpx_null,sizeof(complex));
	memcpy(&layer->L_mx[1][0],&cpx_null,sizeof(complex));
      }
    }
    mlstruct->theta=theta;
  }
  {
    // evaluate the stack matrix
    complex tmp_S[2][2][2];
    int q,r,s,t;
    interface_pars *this_ip;
    this_ip=&(mlstruct->ip[0]);
    
    for (q=0;q<2;q++) {
      for (r=0;r<2;r++) {
	for (s=0;s<2;s++) {
	  if (0) {
	    // prescription for float H_mx
	    memcpy(&mlstruct->S[q][r][s],&cpx_null,sizeof(complex)); // clear
	    mlstruct->S[q][r][s].comp[0]=this_ip->re_H_mx[q][r][s]; // real part
	  } else {
	    // prescription for complex H_mx
	    memcpy(&mlstruct->S[q][r][s],&this_ip->H_mx[q][r][s],sizeof(complex));
	  }
	}
      }
    }
    
    for (i=0;i<mlstruct->nlayer;i++) {

      layer=&(mlstruct->lp[i]);
      // to compute the absorption into each layer we'll need the matrices
      // S1Q (as S1N = S1Q when Q==N) - which is stored in each layer struct
      // copy current S matrix into S1Q matrix.
      for (q=0;q<2;q++) {
	for (r=0;r<2;r++) {
	  for (s=0;s<2;s++) {
	    memcpy(&layer->S1Q[q][r][s],&mlstruct->S[q][r][s],sizeof(complex));
	  }
	}
      }

      if (i < mlstruct->eff_nlayer) {
	for (q=0;q<2;q++) {
	  for (r=0;r<2;r++) {
	    for (s=0;s<2;s++) {
	      memcpy(&tmp_S[q][r][s],&cpx_null,sizeof(complex));
	      memcpy(&mlstruct->S[q][r][s],&cpx_null,sizeof(complex));
	    }
	  }
	}
	for (q=0;q<2;q++) {
	  for (r=0;r<2;r++) {
	    for (s=0;s<2;s++) {
	      for (t=0;t<2;t++) {
		cpx_increment(&tmp_S[q][r][t],
			      cpx_product(&layer->S1Q[q][s][t],
					  &layer->L_mx[s][r]));
	      }
	    }
	  }
	}	
	this_ip=&(mlstruct->ip[i+1]);
	complex cpx_tmp;
	memcpy(&cpx_tmp,&cpx_null,sizeof(complex));
	for (q=0;q<2;q++) {
	  for (r=0;r<2;r++) {
	    for (s=0;s<2;s++) {
	      for (t=0;t<2;t++) {
		if (1) {
		  // float treatment -- apparently this is correct for
		  // absorbing in partially lossy layers. figure this out 
		  // later.. but small errors are showing up when comparing
		  // reflected+transmitted+absorbed (typ. 0.5-3%)
		  cpx_tmp.comp[0]=this_ip->re_H_mx[s][r][t];
		  // the following assignment doesn't work: gets jibberish.
		  // cpx_tmp.comp[0]=this_ip->H_mx[s][r][t].comp[0];
		  // can we do this properly without using the kludge re_H_mx?
		  // a clue may lie with my choice for using cpx_increment
		  // over memcpy.. why does this use cpx_increment? 
		  // gotta check the algorithm!
		  cpx_increment(&mlstruct->S[q][r][t],
				cpx_product(&tmp_S[q][s][t],
					    &cpx_tmp));
		} else {
		  // complex treatment
		  // normalization error referred to above is worse in 
		  // this case.. some problem in the algorithm?
		  // probably should figure out a better way to construct 
		  // mlstruct->S[q][r][t]..?
		  cpx_increment(&mlstruct->S[q][r][t],
				cpx_product(&tmp_S[q][s][t],
					    &this_ip->H_mx[s][r][t]));
		}
	      }
	    }
	  }
	}
      }
    }
    // mlstruct->S now contains the stack matrix, from S1,eff_nlayer.
  }
  {
    // compute the R & T coefficients from complex stack matrix components
    complex inv_s22;
    int s;

    for (s=0;s<2;s++) {
      memcpy(&inv_s22,
	     cpx_inverse(&mlstruct->S[1][1][s]),
	     sizeof(complex));
      memcpy(&mlstruct->rho[s],
	     cpx_product(&inv_s22,&mlstruct->S[0][1][s]),
	     sizeof(complex));

      if (0) {
	// this was called while computing multilayers involving
	// metallic layers. the numeric instability is not yet solved.
	if (mlstruct->eff_nlayer != mlstruct->nlayer) {
	  memcpy(&mlstruct->tau[s],&cpx_null,sizeof(complex));
	} else {
	  memcpy(&mlstruct->tau[s],&inv_s22,sizeof(complex));
	}
      }
      // the old and working solution for dielectric multilayers:
      memcpy(&mlstruct->tau[s],&inv_s22,sizeof(complex));
    }
    // now can compute extinction in each layer
    complex inv_s1q[2][2];
    complex *elq_div_er1;
    complex *erq_div_er1;
    complex *mat[2][2],*inv[2][2];
    int l,m;

    for (i=0;i<mlstruct->nlayer;i++) {
      layer=&(mlstruct->lp[i]);
      for (s=0;s<2;s++) {
	elq_div_er1=&layer->ElQ_div_Er1[s];
	erq_div_er1=&layer->ErQ_div_Er1[s];
	for (l=0;l<2;l++) {
	  for (m=0;m<2;m++) {
	    mat[l][m]=&layer->S1Q[l][m][s];
	    inv[l][m]=&inv_s1q[l][m];
	  }
	}
	cpx_2x2_mat_inv(mat,inv);
	// inv_s1q now contains the inverse of the matrix layer->S1Q
	{
	  //	  display_mat(inv);
	  //	  fprintf(stderr,"\n");
	  
	  complex *E1[2],*Eq[2];
	  complex E[2],E_q[2];

	  for (l=0;l<2;l++) {
	    E1[l] = &E[l];
	    Eq[l] = &E_q[l];
	  }
	  
	  Eq[0]=&layer->ElQ_div_Er1[s];
	  Eq[1]=&layer->ErQ_div_Er1[s];
	  
	  // left going flux in medium 1
	  memcpy(E1[0],&mlstruct->rho[s],sizeof(complex));
	  // right going flux in medium 1
	  E1[1]->comp[0]=1;	  E1[1]->comp[1]=0;
	  
	  for (m=0;m<2;m++) {
	    memcpy(Eq[m],&cpx_null,sizeof(complex));
	    for (l=0;l<2;l++) {
	      memcpy(Eq[m],
		     cpx_increment(Eq[m],cpx_product(&inv_s1q[m][l],E1[l])),
		     sizeof(complex));
	    }
	  }
	  // finished. normalized fluxes in medium q
	  // are stored in layer->ElQ_div_Er1[s].
	}
      }
    }
  }

  float factor=1;
  float atten;
  int s;
  //  fprintf(stderr,"detection in layer (i,d0,d1)=");
  for (i=0;i<mlstruct->nlayer+1;i++) {
    factor *= mlstruct->ip[i].a;
    if (i<mlstruct->nlayer) {
      // use the layer->E{l,r}Q_div_Er1[] etc to compute layer flux stopped
      layer=&(mlstruct->lp[i]);
      atten=exp(-2*layer->beta.comp[1]);
      layer->Dq_ave=0;
      for (s=0;s<2;s++) {
	layer->Dq[s]=factor*(1-atten)*
	  cpx_modulus(&layer->ErQ_div_Er1[s]);
	if (atten>1e-10) {
	  layer->Dq[s]+=factor*(1-atten)*
	    cpx_modulus(&layer->ElQ_div_Er1[s])/atten;
	}
	layer->Dq_ave+=0.5*layer->Dq[s];
      }
    }
  }

  // recalculate factor (computed above) to compute stack transmission etc.
  // repeating this for modularity in the code.

  factor=1;
  for (i=0;i<mlstruct->nlayer+1;i++) factor *= mlstruct->ip[i].a;

  // now can calculate stack transmission using factor.

  mlstruct->R[0]=cpx_modulus(&mlstruct->rho[0]);
  mlstruct->R[1]=cpx_modulus(&mlstruct->rho[1]);

  mlstruct->T[0]=factor*cpx_modulus(&mlstruct->tau[0]);
  mlstruct->T[1]=factor*cpx_modulus(&mlstruct->tau[1]);

  mlstruct->R_ave=0.5*(mlstruct->R[0]+mlstruct->R[1]);
  mlstruct->T_ave=0.5*(mlstruct->T[0]+mlstruct->T[1]);
}

float cpx_modulus (complex *num) {
  complex *p=cpx_product(num,cpx_conjugate(num));
  return(p->comp[0]);
}

char *show_complex (char *s,complex *c) {
  static char st[2048];
  sprintf(st,"%s (r,i)=(%f,%f)\n",s,c->comp[0],c->comp[1]);
  return(st);
}

complex *cpx_increment(complex *a,complex *b) {
  a->comp[0] += b->comp[0];
  a->comp[1] += b->comp[1];
  return(a);
}

complex *cpx_negate(complex *num) {
  static complex cpx_neg;
  cpx_neg.comp[0] = -num->comp[0];
  cpx_neg.comp[1] = -num->comp[1];
  return(&cpx_neg);
}

complex *cpx_conjugate(complex *num) {
  static complex cpx_conj;
  cpx_conj.comp[0] =  num->comp[0];
  cpx_conj.comp[1] = -num->comp[1];
  return(&cpx_conj);
}

complex *cpx_product (complex *a,complex *b) {
  static complex cpx_prod;
  cpx_prod.comp[0] = a->comp[0]*b->comp[0]-a->comp[1]*b->comp[1];
  cpx_prod.comp[1] = a->comp[0]*b->comp[1]+a->comp[1]*b->comp[0];
  return(&cpx_prod);
}

complex *cpx_ratio (complex *a,complex *b) {
  static complex cpx_rat;
  memcpy(&cpx_rat,cpx_product(a,cpx_inverse(b)),sizeof(complex));
  return(&cpx_rat);
}

complex *cpx_inverse (complex *a) {
  static complex cpx_inv;
  float  scalar=1/cpx_modulus(a);
  memcpy(&cpx_inv,cpx_conjugate(a),sizeof(complex));
  cpx_inv.comp[0] *= scalar;  cpx_inv.comp[1] *= scalar;
  return(&cpx_inv);
}

complex *cpx_sqrt (complex *a) {
  static complex cpx_sqroot;
  float sqrtr,phi;
  sqrtr=pow(pow(a->comp[0],2)+pow(a->comp[1],2),0.25);
  phi=atan2(a->comp[1],a->comp[0]);
  cpx_set(&cpx_sqroot,sqrtr*cos(phi/2.0),sqrtr*sin(phi/2.0));
  return(&cpx_sqroot);
}

void init_multilayer(multilayer *ml,
		     int nlayer,float *layer_thickness,optcon *oc,
		     float *oc_const_n) {

  interface_pars **ip=&ml->ip;
  media_pars     **mp=&ml->mp;
  layer_pars     **lp=&ml->lp;

  if (*ip!=NULL) {
    free(*ip);
    *ip=NULL;
  }
  if (*mp!=NULL) {
    free(*mp);
    *mp=NULL;
  }
  if (*lp!=NULL) {
    free(*lp);
    *lp=NULL;
  }

  if (((*ip=(interface_pars*)malloc((nlayer+1)*sizeof(interface_pars)))==NULL) 
      || ((*mp=(media_pars*)malloc((nlayer+2)*sizeof(media_pars)))==NULL)) {
    fprintf(stderr,"mp or ip allocation problem.");
    exit(1);
  }
  
  if (nlayer>0) {
    if ((*lp=(layer_pars*)malloc(nlayer*sizeof(layer_pars)))==NULL) {
      fprintf(stderr,"lp allocation problem.");
      exit(1);
    }
  } else {
    *lp=NULL; // no layers - just an interface.
  }
  
  {
    int i,j;
    for (i=0;i<nlayer+2;i++) {
      (*mp+i)->optical_constants = oc[i];
      (*mp+i)->const_n_val       = oc_const_n[i];
      (*mp+i)->theta=0;
      (*mp+i)->lambda=0;
      (*mp+i)->n.comp[0]=0;
      (*mp+i)->n.comp[1]=0;
    }
    for (i=0;i<nlayer+1;i++) {   // set up the interfaces.
      for (j=0;j<2;j++) (*ip+i)->media_p[j]=(*mp+i+j);
      for (j=0;j<2;j++) (*ip+i)->rho[j]=0;
      for (j=0;j<2;j++) (*ip+i)->tau[j]=0;
      (*ip+i)->a=(*ip+i)->b=0;
    }
    for (i=0;i<nlayer;i++) {     // set up the layers.
      for (j=0;j<2;j++) {
	(*lp+i)->iface_p[j]=(*ip+i+j);
	(*lp+i)->thickness=layer_thickness[i];
      }
    }
  }
  ml->nlayer=nlayer;
  // run once with nonzero angles to initialize
  compute_multilayer(1e-2*rand()/(float)RAND_MAX,
		     123.45+6*rand()/(float)RAND_MAX,
		     ml);
  // ready to return.
}

void vacuum (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  *n_real=1;
  *n_imag=0;
  return;
}

void coating (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  *n_real=1.22;
  *n_imag=0;
  return;
}

void metal (float lambda,complex *n) {
  // uses plasma frequency and damping time.
  // wp = 6e15 rad/s
  // 1/tau = 3e13 rad/s
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  float c=299792458.0*1e9; // nm/s
  float wp=6.0e15;      // rad/s
  float inv_tau=3.0e13; // rad/s
  float w=2*M_PI*c/lambda;
  float Kr,Ki;
  float a=pow(w,2)+pow(inv_tau,2);
  float wp2=pow(wp,2);
  Kr = (a-wp2)/(a);
  Ki = (wp2)/(a*w)*inv_tau;
  double sqrt_sum_k=sqrt(pow(Kr,2)+pow(Ki,2));
  *n_real=0.5*(+Kr+sqrt_sum_k);
  *n_imag=0.5*(-Kr+sqrt_sum_k);
  // this check keeps NANs from showing up in the stack transmissions etc.
  if (*n_real < 1e-5) *n_real=1e-5;
}

void al_metal (float lambda,complex *n) {
  // uses plasma frequency and damping time.
  // wp = 6e15 rad/s
  // 1/tau = 3e13 rad/s
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // using IR fit parameters from ordal et al. 1983
  // wt=6.47e2 cm-1
  // wp=1.19e5 cm-1
  float c=299792458.0*1e9; // nm/s
  float wp     =1.19e5 * 2.9979e10 * 2 * M_PI;  // rad/s
  float inv_tau=6.47e2 * 2.9979e10 * 2 * M_PI;  // rad/s
  float w=2*M_PI*c/lambda;
  float Kr,Ki;
  float a=pow(w,2)+pow(inv_tau,2);
  float wp2=pow(wp,2);
  Kr = (a-wp2)/(a);
  Ki = (wp2)/(a*w)*inv_tau;
  double sqrt_sum_k=sqrt(pow(Kr,2)+pow(Ki,2));
  *n_real=0.5*(+Kr+sqrt_sum_k);
  *n_imag=0.5*(-Kr+sqrt_sum_k);
  // this check keeps NANs from showing up in the stack transmissions etc.
  if (*n_real < 1e-5) *n_real=1e-5;
}

void au_metal (float lambda,complex *n) {
  // uses plasma frequency and damping time.
  // wp = 6e15 rad/s
  // 1/tau = 3e13 rad/s
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // using IR fit parameters from ordal et al. 1983
  // wt=2.16e2 cm-1
  // wp=7.25e4 cm-1
  float c=299792458.0*1e9; // nm/s
  float wp     =7.25e4 * 2.9979e10 * 2 * M_PI;  // rad/s
  float inv_tau=2.16e2 * 2.9979e10 * 2 * M_PI;  // rad/s
  float w=2*M_PI*c/lambda;
  float Kr,Ki;
  float a=pow(w,2)+pow(inv_tau,2);
  float wp2=pow(wp,2);
  Kr = (a-wp2)/(a);
  Ki = (wp2)/(a*w)*inv_tau;
  double sqrt_sum_k=sqrt(pow(Kr,2)+pow(Ki,2));
  *n_real=0.5*(+Kr+sqrt_sum_k);
  *n_imag=0.5*(-Kr+sqrt_sum_k);
  // this check keeps NANs from showing up in the stack transmissions etc.
  if (*n_real < 1e-5) *n_real=1e-5;
}

void ag_metal (float lambda,complex *n) {
  // uses plasma frequency and damping time.
  // wp = 6e15 rad/s
  // 1/tau = 3e13 rad/s
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // using IR fit parameters from ordal et al. 1983
  // wt=1.45e2 cm-1
  // wp=7.25e4 cm-1
  float c=299792458.0*1e9; // nm/s
  float wp     =7.25e4 * 2.9979e10 * 2 * M_PI;  // rad/s
  float inv_tau=1.45e2 * 2.9979e10 * 2 * M_PI;  // rad/s
  float w=2*M_PI*c/lambda;
  float Kr,Ki;
  float a=pow(w,2)+pow(inv_tau,2);
  float wp2=pow(wp,2);
  Kr = (a-wp2)/(a);
  Ki = (wp2)/(a*w)*inv_tau;
  double sqrt_sum_k=sqrt(pow(Kr,2)+pow(Ki,2));
  *n_real=0.5*(+Kr+sqrt_sum_k);
  *n_imag=0.5*(-Kr+sqrt_sum_k);
  // this check keeps NANs from showing up in the stack transmissions etc.
  if (*n_real < 1e-5) *n_real=1e-5;
}

void const_n_func (float lambda,complex *n) {
  //  double *n_real=&n->comp[0];
  //  double *n_imag=&n->comp[1];
  // dummy routine.
  return;
}

void glass (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  *n_imag=0; // for now
  // this was taken from asphere.c for SiO2: cauchy plus a polynomial.
  // lambda is in nm
  *n_real=1.45229948
    +2717.00875*pow(lambda,-2)
    +4.88616329E+7*pow(lambda,-4)
    +lambda*(-1.85896965E-06
	    -2.76150231E-10*lambda);
  return;
}

void air (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  *n_imag=0; // for now
  // taken from asphere.c for air (the one that doesn't depend on 
  // relative humidity or air pressure.
  *n_real=1.000287566
    +1.3412e-18*pow(lambda*1e-9,-2)
    +3.777e-32*pow(lambda*1e-9,-4);
  return;
}

void HfO2(float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // parametrizations from reading off plots by Cerac company
  // (fully oxidized Hafnium Oxide (HFO2)
  // lambda is in nanometers.
  *n_real=1.94509804+12612.7842*pow(lambda,-2)+279157824.*pow(lambda,-4);
  *n_imag=0.000428009633-336.380249*pow(lambda,-2)+43196952.0*pow(lambda,-4);
  if (*n_imag<1e-7 || lambda>450) *n_imag=1e-7;
#ifdef NOLOSSY
  *n_imag=0.0;
#endif 
  return;
}


void SiO2_orig (float lamda,complex *n) {
  glass(lamda,n);
  return;
}

void Si_poly_cpxn (float lambda,complex *n) {
  static complex_n_struct *cpxn=NULL;
  if (cpxn==NULL) {
    cpxn=init_cpx_index("/home/arasmus/ray2_export/Sopra_Data/SIPOLY.cpxn");
  }
  interp_cpx_index(lambda,cpxn,n);
  return;
}

void SiO2 (float lambda,complex *n) {
  static complex_n_struct *cpxn=NULL;
  if (cpxn==NULL) {
    cpxn=init_cpx_index("/home/arasmus/ray2_export/Sopra_Data/SIO2.cpxn");
  }
  interp_cpx_index(lambda,cpxn,n);
  return;
}

void TiO2(float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // this result is from refractiveindex.info allegedly from 
  // handbook of optics (1994)
  double c[]={5.913,0.2441,0.0803};
  *n_real=sqrt(c[0]+c[1]*pow(lambda/1e3,2)/(pow(lambda/1e3,2)-c[2]));
  *n_imag=*n_imag;
#ifdef NOLOSSY
  *n_imag=0.0;
#endif 
  return;
}

void Ta2O5(float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // parametrizations from reading off plots by Cerac company
  // Tantalum pentoxide with/without ion assist (sputtering)
  // lambda is in nanometers.
  *n_real=1.97315955+13501.9453*pow(lambda,-2)+1.58290138e9*pow(lambda,-4);
  // with ion assist sputtering
  *n_real=2.0812428+15724.5205*pow(lambda,-2)+1.80425331e9*pow(lambda,-4);
  *n_imag=0; // dont know any better
  // 0.000428009633-336.380249*pow(lambda,-2)+43196952.0*pow(lambda,-4);
  if (*n_imag<1e-7 || lambda>450) *n_imag=1e-7;
#ifdef NOLOSSY
  *n_imag=0.0;
#endif 
  return;
}

void Si3N4_cpxn(float lambda,complex *n) {
  static complex_n_struct *cpxn=NULL;
  if (cpxn==NULL) {
    cpxn=init_cpx_index("/home/arasmus/ray2_export/Sopra_Data/SI3N4.cpxn");
  }
  interp_cpx_index(lambda,cpxn,n);
}

void Au_cpxn(float lambda,complex *n) {
  static complex_n_struct *cpxn=NULL;
  if (cpxn==NULL) {
    cpxn=init_cpx_index("/home/arasmus/ray2_export/Sopra_Data/AU.cpxn");
  }
  interp_cpx_index(lambda,cpxn,n);
}


void set_ml_Si_Temp(float T) {
  ml_Si_T=T;
}

void Si  (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // cut & pasted from bi_ccd_func.c
  // abs_coeff returns cm-1 with wavelength in Angstroms
  *n_real=n_silicon(10*lambda);
  *n_imag=(lambda*1e-7*abs_coeff(10*lambda,ml_Si_T))/(4*M_PI); 
  return;
}

void silicone_oil  (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  *n_imag=0;
  *n_real=1.53;
  return;
}

void MgF2(float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  // parametrizations from reading off table by Cerac company
  // lambda is in nanometers.
  *n_real=1.44677138+
    lambda*(-0.000266389863+
	    lambda*(3.10344916E-07+
		    lambda*(-1.81796883E-10+
			    lambda*(4.11061193E-14))));
  // Wavelength microns Absorption Coefficient cm-1
  // 2.8 0.0055
  // 5.1 0.006
  // 6.1 0.1
  // very small in the visible range. use the following parametrization.
  *n_imag=0.00549999718+0.224263534*exp(-(lambda-6264.87305)/-190.776276);
#ifdef NOLOSSY
  *n_imag=0.0;
#endif 
  return;
}

void H2O (float lambda,complex *n) {
  double *n_real=&n->comp[0];
  double *n_imag=&n->comp[1];
  {
    // based on fitting data of warren & brandt
    *n_real=1.30655146
      +1.78469932/(1+pow(2*(lambda/1000-0.0718518198)/0.0576468296,2))
      -0.00675903214*pow(lambda/1000,2);
    *n_imag  = 9.59698224E-11*exp((lambda/1000 - 0.656042397)/0.0486679263);
    *n_imag += 5.08359221E-07*exp(-pow((lambda/1000-1.026693)/0.0181379,2)/2);
    *n_imag += 1.61952244E-06*exp(-pow((lambda/1000-1.0365634)/0.05266887,2)/2);
    *n_imag += 1.94868974E-07*exp(-pow((lambda/1000-0.9032985)/0.02546948,2)/2);
    *n_imag += 1.66825458E-07*exp(-pow((lambda/1000-0.882589)/0.0977071,2)/2);
    return;
  }
  {
    // original formulation based on Fernandez-Prini and Dooley etc..
    // committee publication
    float a[]={0.244257733,9.74634476e-3,-3.73234996e-3,2.68678472e-4,
	       1.58920570e-3,2.45934259e-3,0.900704920,-1.66626219e-2};
    
    float rho_star=1000.0; // kg/m3
    float rho=1000.0; //
    float rho_bar=rho/rho_star; // very nearly unity
    
    float T_star=273.15;
    float T=173.15; // -100C kelvin
    T=273.15;
    float T_bar=T/T_star;
    
    float lambda_star=589.0; // nm
    float lam_uv_bar=0.229202; // already divided by lambda_star
    float lam_ir_bar=5.432937; // already divided by lambda_star
    float lam_bar2=pow(lambda/lambda_star,2); // convert to microns
    float lam_uv_bar2=pow(lam_uv_bar,2); // convert to microns
    float lam_ir_bar2=pow(lam_ir_bar,2); // convert to microns
    
    float K;
    K  = a[0];
    K += a[1]*rho_bar;
    K += a[2]*T_bar;
    K += a[3]*lam_bar2*T_bar;
    K += a[4]/lam_bar2;
    K += a[5]/(lam_bar2-lam_uv_bar2);
    K += a[6]/(lam_bar2-lam_ir_bar2);
    K += a[7]*pow(rho_bar,2);
    K *= rho_bar;
    
    *n_real=sqrt((2*K+1)/(1-K));
    fprintf(stderr,"t_bar %f rho_bar %f lam_bar2 %f\n",T_bar,rho_bar,lam_bar2);
    fprintf(stderr,"H2O: *n_real = %f\n",*n_real);
    *n_imag=0;
    return;
  }
}

optcon get_material(char *s) {
  int i;
  for (i=0;i<n_ml_media;i++) {
    if (strcmp(s,ml_media_names[i])==0)
      return(ml_media[i]);
  }
  // if control arrives here there wasn't a match.
  sprintf(err,"no match for material %s",s);
  ml_complain(err);
  return(NULL);
}

void ml_complain(char *s) {
  fprintf(stderr,"%s\n",usage_str);
  fprintf(stderr,"%s\n",s);
  exit(1);
}

complex *cpx_set(complex *c,float re,float im) {
  c->comp[0]=re;
  c->comp[1]=im;
  return(c);
}

complex *cpx_sin(complex *sin_theta,complex *theta) {
  cpx_set(sin_theta,
	  sin(theta->comp[0])*cosh(theta->comp[1]),
	  cos(theta->comp[0])*sinh(theta->comp[1]));
  return(sin_theta);
}

void cpx_2x2_mat_inv  (complex *a[][2], complex *inv[][2]) {
  complex cpx_null,factor,det,tmp_cpx,tmp_inv[2][2];
  cpx_null.comp[0]=cpx_null.comp[1]=0;
  int i,j,k,l;
  cpx_2x2_mat_det(a,&det);
  memcpy(&tmp_cpx,cpx_inverse(&det),sizeof(complex));
  // put inverse of a into tmp_inv
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      // populate tmp_inv with adjoint of a 
      factor.comp[1]=0;
      factor.comp[0]=((i+j)%2==0)?1:-1;
      memcpy(&factor,cpx_product(&factor,&tmp_cpx),sizeof(complex));
      for (k=0;k<2;k++) {
	if (k==i) continue;
	for (l=0;l<2;l++) {
	  if (l==j) continue;
	  // accumulate the cofactor of the element
	  memcpy(&factor,cpx_product(&factor,a[k][l]),sizeof(complex));
	}
      }
      // store in transpose order
      memcpy(&tmp_inv[j][i],&factor,sizeof(complex));
    }
  }
  // done. copy into inv[][] below. do this only after computing the inverse
  // tmp_inv[][] in case inv[][] also points to a[][].
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      memcpy(inv[i][j],&tmp_inv[i][j],sizeof(complex));
    }
  }
}

complex *cpx_2x2_mat_det(complex *a[][2], complex *det) {
  complex cpx_null,cpx_tmp,factor;
  int i,j;
  cpx_null.comp[0]=0;  cpx_null.comp[1]=0;

  memcpy(&cpx_tmp,&cpx_null,sizeof(complex));
  for (i=0;i<2;i++) {
    factor.comp[1]=0;
    factor.comp[0] = (i%2==0) ? 1 : -1;
    for (j=0;j<2;j++) {
      memcpy(&factor,cpx_product(&factor,a[(j)%2][(j+i)%2]),sizeof(complex));
    }
    memcpy(&cpx_tmp,cpx_increment(&cpx_tmp,&factor),sizeof(complex));
  }
  memcpy(det,&cpx_tmp,sizeof(complex));
  return(det);
}

void cpx_2x2_mat_prod(complex *a[][2], complex *b[][2], complex *prod[][2]) {
  // will compute:
  //     prod = a . b
  complex cpx_null,tmp_prod[2][2];
  cpx_null.comp[0]=cpx_null.comp[1]=0;
  int i,j,k;
  
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      memcpy(&tmp_prod[i][j],&cpx_null,sizeof(complex));
      for (k=0;k<2;k++) {
	cpx_increment(&tmp_prod[i][j],cpx_product(a[i][k],b[k][j]));
      }
    }
  }
  // transfer the product (tmp_prod[2][2]) into argument array (*prod[][2])
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      memcpy(prod[i][j],&tmp_prod[i][j],sizeof(complex));
    }
  }
  return;
}

void display_mat(complex *a[][2]) {
  int i,j;
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      fprintf(stderr,"(%f,j*%f)\t",a[i][j]->comp[0],a[i][j]->comp[1]);
    }
    fprintf(stderr,"\n");
  }
}

void compare_float_complex (char *s,float f,complex *c) {
  char str[2048];
  return;
  fprintf(stderr,"%s:float: %f complex: %s\n",s,f,show_complex(str,c));
}
