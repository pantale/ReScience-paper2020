int threads = omp_get_max_threads(); // number of threads
Vector Fint = 0.0; // internal force Vector
Vector FintLoc[threads]; // local internal force vectors
elements.init(); // list of jobs to do

// parallel computation of local internal force vectors
#pragma omp parallel
{
  Element* element;
  while (element = elements.next()) 
    {
      Vector FintElm; // element force vector
      element->computeInternalForces (FintElm);
      FintLoc[omp_get_thread_num()].gatherFrom (FintElm, element);
    }
} // end of parallel loop

// parallel gather operation
#pragma omp parallel for
for (int row = 0; row < Fint.rows(); row++)
{
  for (thread = 0; thread < threads; thread++)
    Fint(row) += FintLoc[thread](row);
} // end of parallel loop
