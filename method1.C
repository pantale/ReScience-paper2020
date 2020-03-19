Vector Fint;

// parallel computation
#pragma omp parallel for
for (int elm = 0; elm < elements.size (); elm++) 
{
  Vector FintElm; // local internal force Vector
  elements(elm).computeInternalForces (FintElm);

#pragma omp critical
  Fint.gatherFrom (FintElm, elements(elm));
} // end of parallel loop
