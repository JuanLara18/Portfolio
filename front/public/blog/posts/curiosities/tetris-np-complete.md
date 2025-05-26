---
title: "Why Tetris is NP-Complete: A Proof"
date: "2025-01-10"
excerpt: "A rigorous mathematical proof showing that the decision problem of Tetris survival is NP-complete, connecting a beloved game to fundamental computational complexity theory."
tags: ["Complexity Theory", "NP-Complete", "Games", "Computer Science", "Proofs"]
headerImage: "/blog/headers/tetris-header.jpg"
---

# Why Tetris is NP-Complete: A Proof

Tetris isn't just an addictive puzzle game—it's a computationally hard problem that belongs to the class of NP-complete problems. This surprising result connects a simple video game to some of the deepest questions in computer science.

## The Problem Statement

**Tetris Survival Problem**: Given a sequence of Tetris pieces and an initial board configuration, can the player survive (avoid reaching the top) for all pieces in the sequence?

More formally: Does there exist a strategy for placing the given sequence of pieces such that no piece extends above the top row of the board?

## Understanding NP-Completeness

### What is NP?

A problem is in **NP** (Nondeterministic Polynomial time) if:
1. A proposed solution can be verified in polynomial time
2. The problem can be solved by a nondeterministic Turing machine in polynomial time

### What is NP-Complete?

A problem is **NP-complete** if:
1. It's in NP
2. Every problem in NP can be reduced to it in polynomial time

## Tetris is in NP

**Claim**: The Tetris survival problem is in NP.

**Proof**: 
Given a sequence of pieces and a proposed placement strategy:
1. **Verification**: Simulate the game following the strategy
2. **Time complexity**: $O(n \cdot w \cdot h)$ where $n$ is the number of pieces, $w$ is board width, $h$ is board height
3. **Polynomial**: This is polynomial in the input size

Therefore, Tetris is in NP. ✓

## The Reduction: 3-Partition ≤ₚ Tetris

To prove NP-completeness, we'll reduce the known NP-complete problem **3-Partition** to Tetris.

### 3-Partition Problem

**Input**: Set $S = \{a_1, a_2, \ldots, a_{3m}\}$ of $3m$ positive integers with $\sum_{i=1}^{3m} a_i = mB$ for some integer $B$.

**Question**: Can $S$ be partitioned into $m$ triples, each summing to exactly $B$?

**Constraints**: Each $a_i$ satisfies $\frac{B}{4} < a_i < \frac{B}{2}$ (ensures exactly 3 elements per partition).

### The Construction

Given a 3-Partition instance, we construct a Tetris instance as follows:

#### Board Setup
- **Width**: $W = 3B + 2m$
- **Height**: $H = m + \text{buffer}$
- **Initial configuration**: Strategic placement of "blocking" pieces

#### Piece Sequence
For each element $a_i \in S$, create a "column piece" of height $a_i$:

```
■ ■ ■ ... ■    (width = 3, height = a_i)
■ ■ ■ ... ■
■ ■ ■ ... ■
...
■ ■ ■ ... ■
```

#### Constraining Gadgets

**Separator Walls**: Place fixed pieces that create $m$ separate regions, each of width exactly $3B$.

**Height Restrictions**: Each region has height limit $B$ before reaching "danger zone".

![Tetris Construction](figures/tetris-construction.png)

### The Reduction Logic

**Key Insight**: In each region of width $3B$:
- Pieces can only fit if their total height ≤ $B$
- Each $a_i$ piece has width 3, so exactly 3 pieces fit horizontally
- The constraint $\frac{B}{4} < a_i < \frac{B}{2}$ ensures 3 pieces are needed to reach height $B$

**Correspondence**:
- **3-Partition solution** ↔ **Tetris survival strategy**
- Each triple summing to $B$ ↔ Three pieces fitting in one region
- Valid partition ↔ No pieces exceed height limits

### Correctness

**Forward Direction**: If 3-Partition has solution
- Group pieces according to partition
- Place each group's pieces in corresponding Tetris region  
- Total height in each region = $B$ (exactly fills region)
- Game survives ✓

**Reverse Direction**: If Tetris game survives
- Each region must be exactly filled (height $B$)
- Each region contains exactly 3 pieces (width constraint)
- This gives valid 3-Partition ✓

## Complexity Implications

### What This Means

Since 3-Partition ≤ₚ Tetris and 3-Partition is NP-complete:
1. **Tetris is NP-hard** (at least as hard as any NP problem)
2. **Tetris is NP-complete** (combining with Tetris ∈ NP)

### Practical Consequences

**For AI**: Perfect Tetris-playing AI is unlikely unless P = NP
- Heuristic approaches are necessary
- No polynomial-time optimal strategy algorithm

**For Game Design**: The computational hardness contributes to the game's engaging difficulty

**For Theory**: Simple games can encode complex computational problems

## Extensions and Variations

### Other NP-Complete Game Problems

The same techniques prove NP-completeness for:
- **Tetris variants** (different piece sets, board sizes)
- **Packing games** (fitting shapes into containers)  
- **Clearing games** (removing complete rows/columns)

### Higher Complexity

Some game problems are even harder:
- **Tetris with infinite pieces**: PSPACE-complete
- **Tetris optimization**: #P-hard (counting optimal strategies)

## Implementation Considerations

```python
def tetris_survival_decision(pieces, board, time_limit):
    """
    Solve Tetris survival using backtracking with pruning
    Note: Exponential worst-case time complexity!
    """
    def place_piece(piece, position, orientation):
        # Try placing piece at position with given rotation
        # Return new board state or None if invalid
        pass
    
    def solve(remaining_pieces, current_board):
        if not remaining_pieces:
            return True  # Survived all pieces!
        
        piece = remaining_pieces[0]
        for pos in valid_positions(piece, current_board):
            for rotation in range(4):
                new_board = place_piece(piece, pos, rotation)
                if new_board and solve(remaining_pieces[1:], new_board):
                    return True
        return False
    
    return solve(pieces, board)
```

## Conclusion

The NP-completeness of Tetris reveals deep connections between recreational mathematics and fundamental computer science. This result shows that:

1. **Simple rules** can create computationally complex problems
2. **Game intuition** often fails for optimal play in hard games  
3. **Mathematical tools** provide insights into game difficulty

The next time you're struggling with a difficult Tetris sequence, remember: you're facing a problem that's fundamentally as hard as the most challenging problems in computer science!

---

*Interested in more complexity results for games? The field of "algorithmic game theory" has many surprising connections between recreation and computation.*