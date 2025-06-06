import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1) Generate a full 9×9 Sudoku grid by simple backtracking + random digit order
# ------------------------------------------------------------------------------

def generate_complete_sudoku():
    """
    Returns a 9×9 list-of-lists, filled with digits 1–9, satisfying standard Sudoku.
    Uses simple backtracking with random digit order.
    """
    grid = [[0]*9 for _ in range(9)]
    
    def is_valid(r, c, val):
        # check row/col/3×3 block
        if any(grid[r][j] == val for j in range(9)): return False
        if any(grid[i][c] == val for i in range(9)): return False
        br, bc = 3*(r//3), 3*(c//3)
        for i in range(br, br+3):
            for j in range(bc, bc+3):
                if grid[i][j] == val:
                    return False
        return True
    
    def backtrack(cell=0):
        if cell == 81:
            return True
        r, c = divmod(cell, 9)
        if grid[r][c] != 0:
            return backtrack(cell+1)
        digits = list(range(1, 10))
        random.shuffle(digits)
        for d in digits:
            if is_valid(r, c, d):
                grid[r][c] = d
                if backtrack(cell+1):
                    return True
                grid[r][c] = 0
        return False
    
    backtrack()
    return grid


# ------------------------------------------------------------------------------
# 2) Partition the 9×9 into cages based on difficulty
# ------------------------------------------------------------------------------

def partition_no_dup_cages(solution_grid, difficulty):
    """
    Returns (cage_id_map, cage_cells) such that:
      • There are approximately `target_cages` clusters (depending on difficulty).
      • No cluster contains two cells whose digits in solution_grid collide.
    
    We maintain a Union-Find where each root tracks the set of digits in that cluster.
    We only merge two neighboring cells if their clusters’ digit‐sets are disjoint.
    """

    # 1) Helper: Union-Find with “digit‐set” tracking
    all_cells = [(r, c) for r in range(9) for c in range(9)]
    parent = {cell: cell for cell in all_cells}
    # For each root, track the set of digits currently in that cluster
    cluster_digits = {cell: {solution_grid[cell[0]][cell[1]]} for cell in all_cells}

    def find(x):
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # Merge smaller‐into‐larger to keep digit‐set updates fast
        if len(cluster_digits[ra]) < len(cluster_digits[rb]):
            ra, rb = rb, ra
        # Now ra has ≥ digits than rb
        parent[rb] = ra
        cluster_digits[ra].update(cluster_digits[rb])
        del cluster_digits[rb]
        return True

    def neighbors(cell):
        r, c = cell
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                yield (nr, nc)

    # 2) Decide target number of cages based on difficulty
    if difficulty <= 1:
        target_cages = random.randint(42, 48)
    elif difficulty <= 2:
        target_cages = random.randint(35, 42)
    elif difficulty <= 3:
        target_cages = random.randint(28, 34)
    elif difficulty <= 4:
        target_cages = random.randint(22, 28)
    else:
        target_cages = random.randint(18, 22)

    # 3) Initially, each cell is its own cluster (so we start with 81 clusters)
    # We'll iteratively try to merge until we hit `target_cages` or no merges possible.
    #
    # Precompute all adjacent‐pairs list so we can attempt random merges.
    neighbor_pairs = []
    for (r, c) in all_cells:
        for (nr, nc) in neighbors((r, c)):
            if (nr, nc) > (r, c):  # avoid duplicates
                neighbor_pairs.append(((r, c), (nr, nc)))

    random.shuffle(neighbor_pairs)

    # Current number of clusters = len(cluster_digits) (each root is a key)
    while len(cluster_digits) > target_cages:
        merged = False

        # Sweep through a shuffled list of neighbor‐pairs,
        # try to merge clusters if their digit‐sets don’t overlap.
        for (cell_a, cell_b) in neighbor_pairs:
            ra, rb = find(cell_a), find(cell_b)
            if ra == rb:
                continue
            # If clusters have no digit in common, merge them
            if cluster_digits[ra].isdisjoint(cluster_digits[rb]):
                union(ra, rb)
                merged = True
                break

        if not merged:
            # No edge can be merged without causing a duplicate. We’re stuck.
            # Break early rather than infinite‐looping. We’ll have more than target_cages.
            break

        # (Optional) Shuffle again to pick a new random merge order next iteration
        random.shuffle(neighbor_pairs)

    # 4) Build cage_cells from the Union-Find structure
    clusters = defaultdict(list)
    for cell in all_cells:
        root = find(cell)
        clusters[root].append(cell)

    # Assign each cluster a letter A, B, C, …
    cage_cells = {}
    for idx, (root, cells) in enumerate(sorted(clusters.items(), key=lambda x: x[0])):
        letter = chr(ord('A') + idx)
        cage_cells[letter] = cells

    # 5) Build a 9×9 cage_id_map for easy lookup
    cage_id_map = [[''] * 9 for _ in range(9)]
    for letter, cells in cage_cells.items():
        for (r, c) in cells:
            cage_id_map[r][c] = letter

    return cage_id_map, cage_cells

# ------------------------------------------------------------------------------
# 3) Given a completed grid + cage_cells, compute target sums
# ------------------------------------------------------------------------------

def compute_cage_sums(solution_grid, cage_cells):
    """
    solution_grid: 9×9 of digits 1–9
    cage_cells: dict(letter → list of (r,c))
    Returns: cage_sums: dict(letter → integer sum)
    """
    cage_sums = {}
    for letter, cells in cage_cells.items():
        s = sum(solution_grid[r][c] for (r,c) in cells)
        cage_sums[letter] = s
    return cage_sums


# ------------------------------------------------------------------------------
# 4) A backtracking solver + counter to check uniqueness (up to 2 solutions)
# ------------------------------------------------------------------------------

def solve_and_count(cage_cells, cage_sums, limit=2):
    """
    Attempts to count how many solutions (≤ limit) exist that satisfy:
      • Standard Sudoku constraints (rows/cols/3×3)
      • Each cage’s sum, with no repeated digit in a cage.
    Stops early if count ≥ limit. Returns a tuple:
        (solution_count, backtrack_calls)
    """

    # 1) Build a fresh 9×9 “cage_id_map” so every cell knows its letter.
    global cage_id_map
    cage_id_map = [[''] * 9 for _ in range(9)]
    for letter, cells in cage_cells.items():
        for (r, c) in cells:
            cage_id_map[r][c] = letter

    # 2) Prepare a 9×9 “grid” of current assignments (0 = empty)
    grid = [[0] * 9 for _ in range(9)]

    # 3) Keep track of which digits are used in each row/col/block
    row_used   = [set() for _ in range(9)]
    col_used   = [set() for _ in range(9)]
    block_used = [set() for _ in range(9)]

    # 4) For each cage, track: (a) current partial sum, (b) how many cells are still empty
    cage_partial_sum = {letter: 0 for letter in cage_cells}
    cage_empty_count = {letter: len(cage_cells[letter]) for letter in cage_cells}

    solution_count = 0
    backtrack_calls = 0

    # 5) Precompute for each cage: the list of its member coordinates
    #    (This is just a local alias; we already have cage_cells)
    cage_members = cage_cells

    # 6) Build a flat list of all 81 positions in row-major order
    all_positions = [(r, c) for r in range(9) for c in range(9)]

    def backtrack(idx=0):
        nonlocal solution_count, backtrack_calls
        backtrack_calls += 1

        # If we’ve already found ≥ limit solutions, stop immediately
        if solution_count >= limit:
            return

        # If we filled all 81 cells, we have a complete solution
        if idx == 81:
            solution_count += 1
            return

        r, c = all_positions[idx]
        letter = cage_id_map[r][c]

        # Try digits 1..9 in this cell
        for d in range(1, 10):
            # 1) Standard Sudoku row/col/block check
            blk = (r // 3) * 3 + (c // 3)
            if (d in row_used[r]
                or d in col_used[c]
                or d in block_used[blk]):
                continue

            # 2) No repeats in the cage
            #    We'll re‐derive which digits are in the cage by scanning cage_member cells:
            already_in_cage = {
                grid[rr][cc]
                for (rr, cc) in cage_members[letter]
                if grid[rr][cc] != 0
            }
            if d in already_in_cage:
                continue

            # 3) Check cage‐sum bounds: if we place “d” now, can the remaining empty cells
            #    in this cage still reach the target sum?
            new_partial = cage_partial_sum[letter] + d
            remaining_cells = cage_empty_count[letter] - 1  # after placing d

            # 3a) If we fill this cell and it's the last empty in its cage,
            #     we MUST match exactly the target sum.
            if remaining_cells == 0:
                if new_partial != cage_sums[letter]:
                    continue
            else:
                # 3b) Otherwise, we only need: new_partial + (smallest possible sum of rem_cells)
                #     ≤ target ≤ new_partial + (largest possible sum of rem_cells)
                #    (Smallest possible digits: 1,2,3,...; largest: 9,8,7,...)
                #    But we must also exclude digits already in cage. So:
                used_digits = already_in_cage | {d}
                k = remaining_cells
                # Build a sorted list of “available” digits for this cage (1..9 minus used)
                avail = [dd for dd in range(1, 10) if dd not in used_digits]
                if len(avail) < k:
                    continue  # not enough candidates left
                avail.sort()
                # Minimum possible sum for k cells = sum of the k smallest in avail
                min_possible = sum(avail[:k])
                # Maximum possible sum for k cells = sum of the k largest in avail
                max_possible = sum(avail[-k:])
                if new_partial + min_possible > cage_sums[letter]:
                    continue
                if new_partial + max_possible < cage_sums[letter]:
                    continue

            # If we passed all checks, place “d” in (r,c)
            grid[r][c] = d
            row_used[r].add(d)
            col_used[c].add(d)
            block_used[blk].add(d)
            prev_partial = cage_partial_sum[letter]
            prev_empty   = cage_empty_count[letter]
            cage_partial_sum[letter]  = new_partial
            cage_empty_count[letter]  = remaining_cells

            # Recurse to the next cell
            backtrack(idx + 1)

            # Undo placement
            grid[r][c] = 0
            row_used[r].remove(d)
            col_used[c].remove(d)
            block_used[blk].remove(d)
            cage_partial_sum[letter]  = prev_partial
            cage_empty_count[letter]  = prev_empty

            if solution_count >= limit:
                return

        # If no digit 1..9 worked, this branch fails and backtracks
        # print(f"Backtracking at idx={idx} (r={r}, c={c}), no valid digit found")
        return

    # Kick off recursion
    backtrack(0)
    return solution_count, backtrack_calls

# ------------------------------------------------------------------------------
# 5) Combine everything: generate until unique solution is found
# ------------------------------------------------------------------------------

def generate_killer(difficulty, max_tries=100):
    """
    Tries up to max_tries times to: 
      1) generate a complete sudoku
      2) partition into cages
      3) compute sums
      4) test uniqueness (exactly 1 solution)
    Returns: puzzle_str (2‐line string), or raises RuntimeError if none found.
    
    puzzle_str format:
      Line 1: 81 characters (row‐major) giving cage ID 'A'.. 
      Line 2: semicolon-separated "ID:sum" pairs (e.g. A:7;B:13;C:4;…)
    """
    for attempt in range(max_tries):
        sol = generate_complete_sudoku()
        cage_id_map_local, cage_cells = partition_no_dup_cages(sol, difficulty)
        cage_sums = compute_cage_sums(sol, cage_cells)
        count, bt_calls = solve_and_count(cage_cells, cage_sums, limit=2)
        print(f"Attempt {attempt+1}: {count} solutions found, {bt_calls} backtrack calls")
        if count == 1:
            # Build string representation
            line1 = ''.join(cage_id_map_local[r][c] for r in range(9) for c in range(9))
            sums_list = [f"{letter}:{cage_sums[letter]}" for letter in sorted(cage_sums)]
            line2 = ';'.join(sums_list)
            return line1 + "\n" + line2
    raise RuntimeError(f"Failed to generate a unique‐solution puzzle in {max_tries} tries")


# ------------------------------------------------------------------------------
# 6) Parser: from puzzle_str → (cage_id_map, cage_sums)
# ------------------------------------------------------------------------------

def parse_puzzle_string(puzzle_str):
    """
    Input: two-line string:
      Line1 = 81 chars of cage IDs
      Line2 = semicolon-separated letter:sum
    Returns:
      cage_id_map (9×9 of letters), cage_cells (letter→list of (r,c)), cage_sums
    """
    lines = puzzle_str.strip().splitlines()
    if len(lines) != 2 or len(lines[0]) != 81:
        raise ValueError("Invalid puzzle string format")
    seq = lines[0]
    cage_id_map = [['']*9 for _ in range(9)]
    for idx, ch in enumerate(seq):
        r, c = divmod(idx, 9)
        cage_id_map[r][c] = ch
    
    cage_cells = defaultdict(list)
    for r in range(9):
        for c in range(9):
            letter = cage_id_map[r][c]
            cage_cells[letter].append((r,c))
    
    cage_sums = {}
    for token in lines[1].split(';'):
        if not token:
            continue
        letter, sval = token.split(':')
        cage_sums[letter] = int(sval)
    
    return cage_id_map, cage_cells, cage_sums


# ------------------------------------------------------------------------------
# 7) Visualization: draw grid + cages + sum‐labels with matplotlib
# ------------------------------------------------------------------------------

def visualize(cage_id_map, cage_sums, figsize=(6,6)):
    """
    cage_id_map: 9×9 array of letters A, B, ...
    cage_sums: dict(letter → int)
    
    Draws the 9×9 grid, thick lines around cage borders, and places each cage’s sum
    in the top-leftmost cell of that cage.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    plt.axis('off')
    
    # Draw the 9×9 light grid
    for i in range(10):
        lw = 1 if i%3 != 0 else 2
        ax.plot([i, i], [0,9], color='black', linewidth=lw)
        ax.plot([0,9], [i,i], color='black', linewidth=lw)
    
    # For each cell, check its right and bottom neighbor. If they differ in cage ID,
    # draw a thick border between them.
    for r in range(9):
        for c in range(9):
            letter = cage_id_map[r][c]
            # RIGHT neighbor
            if c+1 < 9 and cage_id_map[r][c+1] != letter:
                ax.plot([c+1, c+1], [r, r+1], color='black', linewidth=3)
            elif c == 8:
                ax.plot([c+1, c+1], [r, r+1], color='black', linewidth=3)
            # BOTTOM neighbor
            if r+1 < 9 and cage_id_map[r+1][c] != letter:
                ax.plot([c, c+1], [r+1, r+1], color='black', linewidth=3)
            elif r == 8:
                ax.plot([c, c+1], [r+1, r+1], color='black', linewidth=3)
    
    # Place each cage’s sum at the top-left cell of that cage
    # First find top-left-most cell for each letter
    top_left = {}
    for r in range(9):
        for c in range(9):
            letter = cage_id_map[r][c]
            if letter not in top_left or (r, c) < top_left[letter]:
                top_left[letter] = (r, c)
    for letter, (r, c) in top_left.items():
        s = cage_sums.get(letter, None)
        if s is not None:
            ax.text(c+0.05, r+0.20, str(s), fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 8) Example usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: generate a medium‐difficulty puzzle (difficulty = 3)
    puzzle_str = generate_killer(difficulty=5, max_tries=100)
    print("Puzzle string (2 lines):")
    print(puzzle_str)
    
    # Parse it back
    cage_id_map, cage_cells, cage_sums = parse_puzzle_string(puzzle_str)
    
    # Visualize
    visualize(cage_id_map, cage_sums)
