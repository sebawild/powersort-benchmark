# Powersort Competition Benchmark

This benchmark consists of the submissions to [Track A](https://powersort-competition.github.io/PowersortCompetitionWebsite/#/aboutA) of the [Powersort Competition](https://powersort-competition.github.io/PowersortCompetitionWebsite/).

The goal of this repository is to provide a benchmark for run-adaptive sorting methods.  It consists of a dataset of input orderings (permutations)that demonstrates the differences between the [Powersort](https://en.wikipedia.org/wiki/Powersort) merge policy and the orignial [Timsort merge policy](https://en.wikipedia.org/wiki/Timsort#Merge_criteria).


## Background

Timsort and Powersort are adaptive sorting algorithms: both are faster if the input has long presorted areas (“runs”). However, they differ in the merge policy, i.e., the order in which they combine these naturally occurring into longer runs.

Powersort solves this task of finding good merge trees by implicitly solving an optimization problem looking for a nearly optimal binary search tree; Timsort uses local rules based on the lengths of runs.
A much more detailed description of the differences between Timsort and Powersort can be found in [listsort.txt](https://github.com/python/cpython/blob/main/Objects/listsort.txt#L393) in the CPython source code.
A short proof about correctness and mergecost of Powersort appears in the [Multiway Powersort paper](https://www.wild-inter.net/publications/cawley-gelling-nebel-smith-wild-2023).

The task in Track A of the competition was to find orders of input lists, where the merge policy of Timsort and Powersort have (largely) different cost. Each submission is a text file with a single list of elements, separated by commas (a Python list expression), e.g., “[11, 12, 13, 14, 1, 2, 3]”.

For more information and the ranking of submissions, see the [competition website](https://powersort-competition.github.io/PowersortCompetitionWebsite/#/aboutA).


## License

The inputs in this repository are provided as is, without any warranty. By using them, you agree to the [license terms](LICENSE).

The inputs were originally submitted to the Powersort Competition
various authors, in particular

* Daniel Chadwick
* Ziad Ismaili Alaoui
* [Christopher Jefferson](https://github.com/chrisjefferson)
* Thomas Johnson
* Vincent Jugé


