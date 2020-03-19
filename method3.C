int threads = jobs.getMaxThreads(); // number of threads
Vector Fint = 0.0; // internal force Vector
Vector FintLoc[threads]; // local internal force vectors
jobs.init(elements); // list of jobs to do

// parallel computation of local internal force vectors
#pragma omp parallel
{
  Element* element;
  Job* job = jobs.getJob(); // get the job for the thread
  int thread = jobs.getThreadNum(); // get the thread Id
  while (element = job->next()) 
    {
      Vector FintElm; // element force vector
      element->computeInternalForces (FintElm);
      FintLoc[thread].gatherFrom (FintElm, element);
    }
  job->waitOthers(); // compute waiting time
} // end of parallel loop

// parallel gather operation
#pragma omp parallel for
for (int row = 0; row < Fint.rows(); row++)
{
  for (thread = 0; thread < threads; thread++)
    Fint(row) += FintLoc[thread](row);
} // end of parallel loop
