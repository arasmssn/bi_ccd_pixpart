#include "ray.h"
#include "numrec/include/nr.h"
#include "numrec/include/nrutil.h"
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <argz.h>

int ray_util_verbose=1;
extern char* usage_str;
char  err[2048];

void show_ray (ray *r) {
  fprintf(stderr,"k: %f %f %f\tp: %f %f %f\n",
	  r->k.x,r->k.y,r->k.z,r->p.x,r->p.y,r->p.z);
}

char *show_vector (char *s,vec *v) {
  static char str[2048];
  sprintf(str,"%s: (x,y,z) = (%g,%g,%g)",s,v->x,v->y,v->z);
  return(str);
}

int complain (char *s) {
  fprintf(stderr,"%s",usage_str);
  fprintf(stderr,"%s",s);
  exit(1);
}

int ray_surface_collapse (ray *r,double (*func)(vec *v)) {
  /* all surfaces are expressed as f(x,y,z) */
  double l_0,l_1,s,delta_s;
  vec v;
  int n;
  int maxn=500;

  s=0.0;
  v.x=r->p.x+s*r->k.x;    v.y=r->p.y+s*r->k.y;    v.z=r->p.z+s*r->k.z;  
  l_0  = (*func)(&v);
  delta_s=1.0;
  s += delta_s;
  v.x=r->p.x+s*r->k.x;    v.y=r->p.y+s*r->k.y;    v.z=r->p.z+s*r->k.z;  
  l_1  = (*func)(&v);
  n=0;
  while ((fabs(l_1-l_0)>0) && (fabs(l_0-l_1)>SURF_TOLERANCE || n<2) && n<maxn) {
    delta_s = -l_1*delta_s/(l_1-l_0); // undamped search
    //    delta_s = -0.5*l_1*delta_s/(l_1-l_0); // damped search
    s += delta_s;
    if (isnan(s)) {
      fprintf(stderr,"NAN!!! resetting.. \n");
      s=20.0;
    }
    l_0 = l_1;
    v.x = r->p.x + s*r->k.x;
    v.y = r->p.y + s*r->k.y;
    v.z = r->p.z + s*r->k.z;  
    l_1  = (*func)(&v);
    n++;
  }
  /* can't reflect? - no solution found? */
  if (n>maxn) {
    fprintf(stderr,"missing .. l_1 = %f\n",l_1);
    return(0); 
  }
//  if (s<0) return(0); // ray must head backward. skip this ray
  r->p.x += s*r->k.x;
  r->p.y += s*r->k.y;
  r->p.z += s*r->k.z;
  //  r->pl  += s * (r->n) * modulus(&r->k);
  //  fprintf(stderr,"s=%g n=%g mod=%g pl=%g\n",s,r->n,modulus(&r->k),r->pl);
  return(1);
} 

double dot_prod (vec *v1,vec *v2) {
  return( v1->x * v2->x + v1->y * v2->y + v1->z * v2->z);
}


double modulus (vec *v) {
  return(sqrt(dot_prod(v,v)));
}


void cross_prod (vec *v1,vec *v2,vec *vprod) {
  vec tmp;
  tmp.x = v1->y*v2->z - v1->z*v2->y;
  tmp.y = v1->z*v2->x - v1->x*v2->z;
  tmp.z = v1->x*v2->y - v1->y*v2->x;
  cpvec(&tmp,vprod);
  return;
}

void  figure_error (vec *v,float err1,float err2) {
  // add a bit of error to the vector v
  vec tmp,tmp1,tmp2;
  float xc,yc,zc;

  tmp.x=tmp.y=tmp.z=0;  tmp.x=1;  xc=dot_prod(&tmp,v);
  tmp.x=tmp.y=tmp.z=0;  tmp.y=1;  yc=dot_prod(&tmp,v);
  tmp.x=tmp.y=tmp.z=0;  tmp.z=1;  zc=dot_prod(&tmp,v);

  if (fabs(xc)<fabs(yc)) {
    if (fabs(xc)<fabs(zc)) {      tmp.x=tmp.y=tmp.z=0;  tmp.x=1;  } 
    else                   {      tmp.x=tmp.y=tmp.z=0;  tmp.z=1;  }
  } else {
    if (fabs(yc)<fabs(zc)) {      tmp.x=tmp.y=tmp.z=0;  tmp.y=1;  } 
    else                   {      tmp.x=tmp.y=tmp.z=0;  tmp.z=1;  }
  }

  {
    double modulus;

    cross_prod(&tmp,v,&tmp1);
    modulus=sqrt(dot_prod(&tmp1,&tmp1));
    tmp1.x /= modulus;    tmp1.y /= modulus;    tmp1.z /= modulus;
    cross_prod(&tmp1,v,&tmp2);
    modulus=sqrt(dot_prod(&tmp2,&tmp2));
    tmp2.x /= modulus;    tmp2.y /= modulus;    tmp2.z /= modulus;
    v->x += err1*tmp1.x + err2*tmp2.x;
    v->y += err1*tmp1.y + err2*tmp2.y;
    v->z += err1*tmp1.z + err2*tmp2.z;
    modulus=sqrt(dot_prod(v,v));
    v->x /= modulus;    
    v->y /= modulus;    
    v->z /= modulus;    
  }
}

int surface_normal (vec *v,double (*func)(vec *v),vec *n) {
  vec t1,t2;

  // this routine could be improved by adapting the 
  // gradient finding algorithm to the approximate 
  // orientation of the surface. instead, the gradient vector
  // components are always found in this cartesian way.

  t1.x = v->x + 0.5*SURF_DELT; t1.y = v->y; t1.z = v->z;
  t2.x = v->x - 0.5*SURF_DELT; t2.y = v->y; t2.z = v->z;
  n->x = ((*func)(&t1) - (*func)(&t2))/SURF_DELT;

  t1.x = v->x; t1.y = v->y + 0.5*SURF_DELT; t1.z = v->z;
  t2.x = v->x; t2.y = v->y - 0.5*SURF_DELT; t2.z = v->z;
  n->y = ((*func)(&t1) - (*func)(&t2))/SURF_DELT;

  t1.x = v->x; t1.y = v->y; t1.z = v->z + 0.5*SURF_DELT;
  t2.x = v->x; t2.y = v->y; t2.z = v->z - 0.5*SURF_DELT;
  n->z = ((*func)(&t1) - (*func)(&t2))/SURF_DELT;

  unitvec(n);

  // second pass.. 
  // compute the local normal using directional vectors 
  // where one is nearly aligned with n.
  
  {
    vec x={1.0,0.0,0.0},y={0.0,1.0,0.0},z={0.0,0.0,1.0};
    vec tan1,tan2;

    if (1) {
      // also use the fact that the optical axis is closest aligned to Z.
      cpvec(&z,&tan1);
    } else {
      double nx,ny,nz;
    
      nx = fabs(dot_prod(n,&x));
      ny = fabs(dot_prod(n,&y));
      nz = fabs(dot_prod(n,&z));
      
      if ((nx<ny) && (nx<nz))      {      cpvec(&x,&tan1);    } 
      else if ((ny<nx) && (ny<nz)) {      cpvec(&y,&tan1);    } 
      else                         {      cpvec(&z,&tan1);    }
    }

    cross_prod(n,&tan1,&tan2);    cross_prod(&tan2,n,&tan1);
    unitvec(&tan1);               unitvec(&tan2);
    // n, tan1 and tan2 are now orthogonal.
    {
      vec n1,n2,o1,o2,p1,p2;
      double spc=SURF_DELT;

      cpvec(n,&n1);           cpvec(n,&n2);
      cpvec(&tan1,&o1);       cpvec(&tan1,&o2);
      cpvec(&tan2,&p1);       cpvec(&tan2,&p2);
      scalevec(&n1,0.5*spc);  scalevec(&n2,-0.5*spc);
      scalevec(&o1,0.5*spc);  scalevec(&o2,-0.5*spc);
      scalevec(&p1,0.5*spc);  scalevec(&p2,-0.5*spc);
      vec_add(v,&n1,&n1);     vec_add(v,&n2,&n2);
      vec_add(v,&o1,&o1);     vec_add(v,&o2,&o2);
      vec_add(v,&p1,&p1);     vec_add(v,&p2,&p2);
      scalevec(n,    ((*func)(&n1) - (*func)(&n2))/spc);
      scalevec(&tan1,((*func)(&o1) - (*func)(&o2))/spc);
      scalevec(&tan2,((*func)(&p1) - (*func)(&p2))/spc);
      vec_add(n,&tan1,n);     vec_add(n,&tan2,n);
      unitvec(n);
    }
  }
  //  fprintf(stderr,"%s\n",show_vector("norm:",n));
  return(1);
}

int refract_ray(vec *k,vec *normal,float n_from,float n_to) {
  vec kn,nc;

  cpvec(k,&kn);  unitvec(&kn);
  cpvec(normal,&nc);  scalevec(&nc,dot_prod(&kn,normal));
  vec_diff(&kn,&nc,&kn);
  scalevec(&kn,n_from/n_to);
  if (modulus(&kn)<1) {
    unitvec(&nc);
    scalevec(&nc,sqrt(1-dot_prod(&kn,&kn)));
    vec_add(&nc,&kn,&kn);
    scalevec(&kn,modulus(k));
    cpvec(&kn,k);
    return(1);
  } else {
    // ray is swallowed (total internal reflection)
    return(0);
  }
}

int reflect_ray(vec *k,vec *surf_norm) {
  float k_dot_n;
  k_dot_n=dot_prod(k,surf_norm);
  k->x -= 2 * k_dot_n * surf_norm->x;
  k->y -= 2 * k_dot_n * surf_norm->y;
  k->z -= 2 * k_dot_n * surf_norm->z;
  return(1);
}

void vec_diff (vec *v1,vec *v2,vec *vecdif) {
  vec tmp;
  tmp.x=v1->x-v2->x;
  tmp.y=v1->y-v2->y;
  tmp.z=v1->z-v2->z;
  cpvec(&tmp,vecdif);
}

void vec_add (vec *v1,vec *v2,vec *vecsum) {
  vec tmp;
  tmp.x=v1->x+v2->x;
  tmp.y=v1->y+v2->y;
  tmp.z=v1->z+v2->z;
  cpvec(&tmp,vecsum);
}


void unitvec (vec *v) {
  double modulus=sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
  if (modulus==0) {
    // if this is a zero length vector, return a zero length unit vector.
    v->x = 0;    v->y = 0;    v->z = 0;
  } else {
    v->x /= modulus;  v->y /= modulus;  v->z /= modulus;
  }
  return;
}

void scalevec (vec *v,float s) {
  v->x *= s;   v->y *= s;   v->z *= s; 
  return;
}

void cpvec (vec *vfrom,vec *vto) {
  vto->x=vfrom->x;
  vto->y=vfrom->y;
  vto->z=vfrom->z;
  return;
}


void tran_ray(ray *aray,lin_trans *tf,int direction) {
  double pv[3],kv[3],p1v[3],k1v[3];
  int i,j;

  pv[0]=aray->p.x;
  pv[1]=aray->p.y;
  pv[2]=aray->p.z;
  kv[0]=aray->k.x;
  kv[1]=aray->k.y;
  kv[2]=aray->k.z;

  if (tf->reverse) {
    if (direction>0) {
      // do nothing
    } else {
      pv[0]+=tf->v_offset.x;
      pv[1]+=tf->v_offset.y;
      pv[2]+=tf->v_offset.z;
    }
  } else {
    if (direction>0) {
      pv[0]-=tf->v_offset.x;
      pv[1]-=tf->v_offset.y;
      pv[2]-=tf->v_offset.z;
    } else {
      // do nothing
    }
  }
  
  // rotate p and k components.
  
  if (direction>0) {
    for (j=0;j<3;j++) {
      p1v[j]=0.0;
      k1v[j]=0.0;
      for (i=0;i<3;i++) {
	p1v[j] += tf->R_tot[j][i]*pv[i];
	k1v[j] += tf->R_tot[j][i]*kv[i];
      }
    }
  } else {
    for (j=0;j<3;j++) {
      p1v[j]=0.0;
      k1v[j]=0.0;
      for (i=0;i<3;i++) {
	p1v[j] += tf->R_tot[i][j]*pv[i];
	k1v[j] += tf->R_tot[i][j]*kv[i];
      }
    }
  }

  aray->p.x = (float) p1v[0];
  aray->p.y = (float) p1v[1];
  aray->p.z = (float) p1v[2];
  aray->k.x = (float) k1v[0];
  aray->k.y = (float) k1v[1];
  aray->k.z = (float) k1v[2];

  if (tf->reverse) {
    if (direction>0) {
      aray->p.x-=tf->v_offset.x;
      aray->p.y-=tf->v_offset.y;
      aray->p.z-=tf->v_offset.z;
    } else {
      // do nothing
    }
  } else {
    if (direction>0) {
      // do nothing
    } else {
      aray->p.x+=tf->v_offset.x;
      aray->p.y+=tf->v_offset.y;
      aray->p.z+=tf->v_offset.z;
    }
  }
  return;
}

void parse_transform_subargs(char *transform_string,int *argc,char **argv[]) {
  // routine to take out of (argc,argv) the parts that form a transformation 
  // string. (collect arguments into a single string until '--' is encountered.
  sprintf(transform_string,""); // initialize, same as transform_string[0]=0;
  do {
    if (((*argv)[0][0]=='-') && ((*argv)[0][1]=='T')) 
      goto nextarg;
    if (((*argv)[0][0]=='-') && ((*argv)[0][1]=='-')) {
      // finished. pass control back to calling program.
      return;
    } else {
      sprintf(transform_string,"%s %s",transform_string,(*argv)[0]);
    }
  nextarg:
    (*argv)++;
  } while (--(*argc));
  return;
}

void fill_transform_specs (char *ts,lin_trans *tf) {
  char *argz,*entry=NULL;
  int  argc;
  size_t  argz_len;
  double tx,ty,tz,rx,ry,rz;
  int    degrees=0;

  tx=ty=tz=0.0;
  rx=ry=rz=0.0;
  tf->reverse=0;
  tf->invert=0;
  if (argz_create_sep(ts,' ',&argz,&argz_len)) {
    fprintf(stderr,"argz_create failure! (in fill_transform_specs)\n");
    exit(1);
  }
  argc=argz_count(argz,argz_len);
  entry=NULL;
  while (argc--) {
    entry=argz_next(argz,argz_len,entry);
    switch (entry[0]) {
    case '-':
      switch(entry[1]) {
      case 't': 
	if (argc<3) complain("arguments expected after -t\n");
	--argc;	entry=argz_next(argz,argz_len,entry);
	tx=atof(entry);
	--argc;	entry=argz_next(argz,argz_len,entry);
	ty=atof(entry);
	--argc;	entry=argz_next(argz,argz_len,entry);
	tz=atof(entry);
	break;
      case 'r': 
	if (argc<3) complain("arguments expected after -r\n");
	--argc;	entry=argz_next(argz,argz_len,entry);
	rx=atof(entry);
	--argc;	entry=argz_next(argz,argz_len,entry);
	ry=atof(entry);
	--argc;	entry=argz_next(argz,argz_len,entry);
	rz=atof(entry);
	break;
      case 'R':
	tf->reverse=1;
	break;
      case 'I':
	tf->invert=1;
	break;
      case 'd':
	degrees=1;
	break;
      default:
	sprintf(err,"unknown switch: %s\n",entry);
	complain(err);
	break;
      }
      break;
    case 0:
      break;
    default:
      fprintf(stderr,"can't parse: (%s) something's wrong with transform specs: %s\nmust be compatible with tran_ray\n",entry,ts);
      exit(1);
      break;
    }
  }

  tf->v_offset.x=tx;  tf->v_offset.y=ty;  tf->v_offset.z=tz;

  if (degrees) {
    // rx,ry & rz arguments were in degrees. convert to radians.
    rx *= M_PI/180.0;    ry *= M_PI/180.0;    rz *= M_PI/180.0;
  }


  // prepare the matrices.
  {
    double Rxi[3][3],Ryi[3][3],Rzi[3][3];
    int i,j,k1,k2;
    
    // i is row number
    // j is column number
    
    for (i=0;i<3;i++) 
      for (j=0;j<3;j++)
	switch((i-j+3)%3) {
	case 0:
	  Rxi[i][j]=(j==0)?1:cos(rx);
	  Ryi[i][j]=(j==1)?1:cos(ry);
	  Rzi[i][j]=(j==2)?1:cos(rz);
	  break;
	case 1:
	  Rxi[i][j] = ((i-0)*(j-0)==0) ? 0.0 : -sin(rx);
	  Ryi[i][j] = ((i-1)*(j-1)==0) ? 0.0 : -sin(ry);
	  Rzi[i][j] = ((i-2)*(j-2)==0) ? 0.0 : -sin(rz);
	  break;
	case 2:
	  Rxi[i][j] = ((i-0)*(j-0)==0) ? 0.0 : +sin(rx);
	  Ryi[i][j] = ((i-1)*(j-1)==0) ? 0.0 : +sin(ry);
	  Rzi[i][j] = ((i-2)*(j-2)==0) ? 0.0 : +sin(rz);
	  break;
	default:
	  break;
	}
    // multiply the matrices: tf->R_tot = Rxi Ryi Rzi - this is a 3-2-1 rotation.
    for (i=0;i<3;i++)
      for (j=0;j<3;j++) {
	tf->R_tot[i][j]=0.0;
	for (k1=0;k1<3;k1++)
	  for (k2=0;k2<3;k2++)
	    tf->R_tot[i][j]+=Rxi[i][k1]*Ryi[k1][k2]*Rzi[k2][j];	    
      }
    
    if (ray_util_verbose==1) {
      fprintf(stderr,"translation vector to be used:\n");
      if (tf->reverse) {
	fprintf(stderr,"\t(translation performed after rotation - reversed from usual)\n");      
      }
      fprintf(stderr,"%16.10g %16.10g %16.10g\n\n",tx,ty,tz);
      fprintf(stderr,"rotation matrix to be used:\n");
      fprintf(stderr,"%16.10g %16.10g %16.10g\n",tf->R_tot[0][0],tf->R_tot[0][1],tf->R_tot[0][2]);
      fprintf(stderr,"%16.10g %16.10g %16.10g\n",tf->R_tot[1][0],tf->R_tot[1][1],tf->R_tot[1][2]);
      fprintf(stderr,"%16.10g %16.10g %16.10g\n",tf->R_tot[2][0],tf->R_tot[2][1],tf->R_tot[2][2]);
      
    }
  }
}
