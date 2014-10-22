#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "ray.h"
#include "field_superpose.h"

typedef struct field_contrib_mgmt {
  vec *contribs;
  int ncontrib;
  int n_contrib_chunk;
} field_contrib_mgmt;

int comp_vec_modulus(const void *a,const void *b);
int comp_double_modulus(const void *a,const void *b);

void multipole_stack_init (multipole_stack **ms) {
  // take the multipole stack and free up elements 
  // of the linked list it points to

  multipole_stack *msi=*ms;
  multipole_stack *ms_temp;

  while (msi != NULL) {
    if (msi->m_element != NULL) free(msi->m_element);
    ms_temp=msi;
    msi=msi->ms_next;
    if (ms_temp != NULL) free(ms_temp);
  }
  *ms=NULL;
  // return
}

void multipole_stack_append (
    multipole_stack **ms,
    void (*fnc)(vec *x,vec *p,vec *x0,vec *sym_axis,vec *field),
    vec *p,vec *x0,vec *sym_axis) {

  multipole_stack **msi=ms;
  while (*msi != NULL) {
    msi=&(*msi)->ms_next;
  }
  *msi=(multipole_stack*)malloc(sizeof(multipole_stack));
  (*msi)->m_element=(multipole_element*)malloc(sizeof(multipole_element));
  (*msi)->m_element->func=fnc;
  
  memcpy(&(*msi)->m_element->p,p,sizeof(vec));
  memcpy(&(*msi)->m_element->x0,x0,sizeof(vec));
  if (sym_axis == NULL) {
    vec dummy={0,0,1};
    memcpy(&(*msi)->m_element->sym_axis,&dummy,sizeof(vec));
  } else {
    memcpy(&(*msi)->m_element->sym_axis,sym_axis,sizeof(vec));
  }
  (*msi)->ms_next=NULL;
  // ready to return
}

void multipole_stack_eval (multipole_stack *ms,vec *x,vec *field) {
  multipole_stack *msi=ms;
  vec field_contrib;
  multipole_element *elem;

  // count the elements
  int n_contrib=0;
  while (msi != NULL) {
    msi=msi->ms_next;
    n_contrib++;
  }

  // allocate the list of vector components
  double *xc=(double*)malloc(n_contrib*sizeof(double));
  double *yc=(double*)malloc(n_contrib*sizeof(double));
  double *zc=(double*)malloc(n_contrib*sizeof(double));

  if ((xc==NULL) || (yc==NULL) || (zc==NULL)) {
    // graceful exit
    fprintf(stderr,
	    "can't allocate %d vectors in multipole_stack_eval\nexiting\n",
	    n_contrib);
  }

  msi=ms;
  int i;
  for (i=0;i<n_contrib;i++) {
    elem=msi->m_element;
    elem->func(x,&elem->p,&elem->x0,&elem->sym_axis,&field_contrib);
    xc[i] = field_contrib.x; 
    yc[i] = field_contrib.y;
    zc[i] = field_contrib.z;
    msi=msi->ms_next;
  }

  // now sort vector component contribution lists xc,yc,zc to reduce
  // roundoff errors on summation
  qsort(xc,n_contrib,sizeof(double),comp_double_modulus);
  qsort(yc,n_contrib,sizeof(double),comp_double_modulus);
  qsort(zc,n_contrib,sizeof(double),comp_double_modulus);

  // compute the grand total by size order
  for (i=1;i<n_contrib;i++) {
    xc[0] += xc[i];    yc[0] += yc[i];    zc[0] += zc[i];
  }
  field->x += xc[0];  field->y += yc[0];  field->z += zc[0];
  free(xc);  free(yc);  free(zc);
}

void multipole_stack_inspect (multipole_stack *ms) {
  multipole_stack *msi=ms;
  fprintf(stderr,"multipole_stack head: %p\n",msi);
  while (msi!=NULL) {
    fprintf(stderr,"element %p:\n",msi);
    fprintf(stderr,"\tfunc:\t%p\n",msi->m_element->func);
    fprintf(stderr,"%s\n",show_vector("\tp:\t",&msi->m_element->p));
    fprintf(stderr,"%s\n",show_vector("\tx0:\t",&msi->m_element->x0));
    fprintf(stderr,"%s\n",show_vector("\tsym:\t",&msi->m_element->sym_axis));
    msi=msi->ms_next;
  }
}

int comp_vec_modulus(const void *a,const void *b) {
  double mod1,mod2;
  mod1=modulus((vec*)a);
  mod2=modulus((vec*)b);
  if (mod1<mod2) return(-1);
  if (mod1>mod2) return(1);
  return(0);
}

int comp_double_modulus(const void *a,const void *b) {
  double am = (*(double*)a)*(*(double*)a);
  double bm = (*(double*)b)*(*(double*)b);
  if (am<bm) return(-1);
  if (am>bm) return(+1);
  return(0);
}
