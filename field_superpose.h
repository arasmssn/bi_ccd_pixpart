#ifndef _RAY_H
#include "ray.h"
#endif

typedef void (*field_contrib_func)(vec *x,vec *p,vec *x0,vec *sym_axis,vec *field);

typedef struct multipole_element {
  field_contrib_func func;
  vec p;     // dipole - each func has its own
  vec x0;    // location of dipole- each func has its own
  vec sym_axis; // mostly useful for 2D dipoles etc.. ignored in 3D
} multipole_element;

typedef struct multipole_stack {
  multipole_element *m_element;
  struct multipole_stack *ms_next;
} multipole_stack;

void multipole_stack_eval (multipole_stack *ms,vec *x,vec *field);
void multipole_stack_init (multipole_stack **ms);
void multipole_stack_append (multipole_stack **ms,
	     void (*fnc)(vec *x,vec *p,vec *x0,vec *sym_axis,vec *field),
	     vec *p,vec *x0,vec *sym_axis);
void multipole_stack_inspect (multipole_stack *ms);
