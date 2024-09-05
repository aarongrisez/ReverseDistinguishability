# Commentary

Included here is a brief commentary on the lessons I learned through this project as well as changes I would implement, were this a project I approached today. Everywhere I refer to "the code" below, I mean "the code as it was when the project was complete". This most directly aligns with this repository's contents as of commit 4580519e35ce1e066a764d13b92ac0e89bb63ad7.

## Documentation

### Quantity Definitions
Many quantities were under investigation during this project. It is difficult to look at the code and understand the meaning of a particular quantity. Specific questions that are relevant to the quantities involved in this project include:

1. Which of the following objects does the quantity describe?
    a. A single quantum system
    b. A pair of quantum systems
    c. The tensor product of one quantum system with itself N times
    d. The tensor product of a pair of quantum systems with themselves N times
    e. The limit of a quantity of type (c) as N -> infinity
    f. The limit of a quantity of type (d) as N -> infinity
    g. The maximum or minimum of any of the above quantities when parameterized by 1 or more external parameters
2. Is a closed form expression for calculating the quantity known? If so, what is it?
3. If not (2), how can the expression be calculated?
4. Narratively, what does this quantity represent (if possible to describe)? If not, why?

Some example quantities that would have benefited from a concise accounting of answers to the above questions include:

- Quantum Chernoff Bound
- Quantum Relative Entropy
- Quantum Reverse Chernoff Bound
- Quantum Reverse Relative Entropy
- Matsumoto Quantity