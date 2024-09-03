# Reverse Distinguishability

## Narrative Background
During the Summer of 2018, I spent some time at the [Perimeter Institute for Theoretical Physics](https://perimeterinstitute.ca/) in Waterloo, Ontario. This repository houses code related my research project, set by [Matt Leifer](https://www.chapman.edu/our-faculty/matt-leifer). Much of the code has remained untouched.

I leave this repository on my public GitHub profile for a variety of reasons. First and foremost, this project was integral to my professional development. While I didn't end up pursuing an academic career in physics, I still gained a lot through this project. Some specific highlights include:
- Exposure to Linear and Semidefinite programming
- Working inside extreme ambiguity
- Learning the fundamentals of Information Theory

## Technical Background
There are several broad topic areas of interest relevant to this project. The following is a high-level description of each topic. While it attempts to be as precise as possible, it is meant more as an accessible overview for those not familiar with the field.

### Distinguishability of Probability Distributions
Imagine you have access to a loaded die which exhibits *one* of the following behaviors:
a. Rolls a "one" 50% of the time and everything else equally.
b. Rolls a "two" 50% of the time and everything else equally.

Given the opportunity to roll this die an unlimited number of times, how might you distinguish whether the die is of type (a) or type (b)? This problem is a specific example of hypothesis testing and has been studied extensively.

We can ask a secondary question though: how easy is it to distinguish between (a) and (b)? For example, if the biases toward rolling a "one" and "two" were 100% (that is, a die of type (a) always rolls "one" and a die of type (b) always rolls "two"), the problem is trivial: roll once, if the result is "one", your die is of type (a), else the result is "two" and your die is of type (b). Conversely, if the two types of die were nearly fair--say, only a bias of ~1% toward a given roll--then significantly more rolls are required to distinguish the two.

The desire to quantify how "easy" it is to distinguish these dice leads directly to the notion of a measure of distinguishability. Very roughly, this type of measurement captures the notion of how much two probability distributions "overlap". There is no singular way of measuring this quantity, and thus there exist an entire family of distinguishability measures.

### Lifting to Quantum
The same story that we told for classical probability distributions can be applied to quantum states. The differences are typical of the lift from Classical to Quantum (probability distributions become density operators, states are projected onto a basis during measurement, etc.)

## Goals
This project built on prior work done by Matt Leifer and Ryan Morris. Broadly, we hoped to identify characteristics of the "reverse variational distance", found in prior work to be a solution to a semidefinite optimization problem. Due to a lemma proved by Leifer and Morris, this measure could be used as a lower bound on a broad class of quantum distinguishability measures. Therefore, understanding the qualitative behavior of the semidefinite optimization problem was of great interest.

## To Do
- Development Setup instructions