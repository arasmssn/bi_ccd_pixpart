#include <math.h>

float expdev(int *idum)
//int *idum;
{
	float dum,ran1();

	do
		dum=ran1(idum);
	while (dum == 0.0);
	return -log(dum);
}
