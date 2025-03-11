# NeuralOptimizerSearch

All perfomred in Tensorflow

## FinalOptimizers

Contains implementations of the final best 10 Optimizers, Adam Clip variants, and Evolvd Learning Rate Schedules

Bottom of the file contains example on how to use

## NeuralOptimizerSearch

This is the main file that performs the evolution of the Optimizers.

Each optimizer is represented by a chromosome: _OptimizerChrom_. 

Each _OptimizerChrom_ contains a list of _Nodes_ and an Adjacency matrix containing the indices of where each node points to. Another adjacency matrix is also present, this time containing _DecaySchedules_ that are applied to each connection of the adjacency matrix of nodes. The purpose of each Node is to contain what operation it is to perform between its inputs. 

Each _DecaySchedule_ follows a similar flow, containing a small adjacency matrix and list of _DecayNodes_. Each DecayNode contains what operation/decay schedule to perform.

Each optimizer chromosome is passed around until the fitness function is reached. Once the OptimizerChrom is to be evaluated, an Optimizer tensorflow object is created. Note, this implementation is highly convoluted and computes all possible combinations, even if it is not used by the OptimizerChrom.

Note: In code, _un_bin_perc_ is short for unary_binary_percent (the probability of being either unary or binary).

The logic for all ofthis is kind of difficult so I'm sorry if it's hard to understand by reading code. 

## nas_optimizer_particle_n50_k5_m7.log

This is an example log file from an actual run, it is decomposed into two major parts, the Initial population and the actual particle evolution, search for either "TRAINING INITIAL POPULATION" or "INIT PARTICLE 1/50"

Each line will print the Validation Accuracy, the Time taken, and the "Function" (Optimizer implementation).

For example, on line 589 we get the $k=4$ mutation applied at timestep $t=1$ of the $n=1$ particle, 

"None ~ (10-4w)+(min(v hat, [min(a12, a9)][2])) val: 0.8029999732971191 - Time: 202.04096484184265"

This optimizer took 202seconds to train, yielding an 80% validation accuracy. The optimizer implementation used no momentum with the update being $U=10^{-4}w+min(\hat v, [min(a12, a9)][2])$, where $[min(a12, a9)][2]$ is a decay schedule of the form $min(a12,a9)$ applied to the value of $2.0$.
