# Computer-inteligence SOM
self-organizing map in C++

The MNIST data set of handwritten digits

1. Initialization: Choose random values for the initial weight vectors wj(0).
From 0 to i where i is the number of neurons in the lattice. The magnitude
of the weight was kept small.
2. Sampling: Draw a sample x from the input space with a certain probability;
the vector x represents the activation pattern that is applied to the lattice.
The dimension of vector x is equal to m.
3. Similarity matching: Find the best-matching (winning) neuron i(x) at time-
step n by using the minimum-distance criterion
4. Updating: Adjust the synaptic-weight vectors of all excited neurons by using
the update formula
5. Continuation: continue with step 2 until no noticeable changes in the
feature map are observed
