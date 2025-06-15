def compute_score(data_source: str, solution_str, ground_truth, extra_info=None):
    if data_source in ["HuggingFaceH4/MATH-500", "MathArena/aime_2025", "HuggingFaceH4/aime_2024"]:
        from recipe.r1.tasks import math

        return math.compute_score(solution_str, ground_truth)
    elif 'jackcai1206/sudoku_easy2hard' in data_source:
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        
        def parse_sudoku_answer(solution_str):
            try:
                boxed_content = last_boxed_only_string(solution_str)
                if not boxed_content:
                    return None
                
                cleaned = boxed_content.replace('[', '').replace(']', '').replace(',', '').replace(' ', '')
                digits = [int(c) for c in cleaned if c.isdigit()]
                
                if len(digits) == 81:
                    return digits
                return None
            except:
                return None
        
        def compute_sudoku_score(grid, original_puzzle=None):
            if len(grid) != 81:
                return 0.0
            
            # Verify original clues are preserved
            if original_puzzle is not None:
                for i in range(81):
                    if original_puzzle[i] != 0 and grid[i] != original_puzzle[i]:
                        return 0.0
            
            sudoku = []
            for i in range(9):
                row = grid[i*9:(i+1)*9]
                sudoku.append(row)
            
            total_constraints = 27  # 9 rows + 9 columns + 9 boxes
            satisfied_constraints = 0
            
            for num in grid:
                if num < 1 or num > 9:
                    return 0.0
            
            for row in sudoku:
                if sorted(row) == list(range(1, 10)):
                    satisfied_constraints += 1
            
            for col in range(9):
                column = [sudoku[row][col] for row in range(9)]
                if sorted(column) == list(range(1, 10)):
                    satisfied_constraints += 1
            
            for box_row in range(3):
                for box_col in range(3):
                    box = []
                    for i in range(3):
                        for j in range(3):
                            box.append(sudoku[box_row*3 + i][box_col*3 + j])
                    if sorted(box) == list(range(1, 10)):
                        satisfied_constraints += 1
            
            if satisfied_constraints == total_constraints:
                return 1.0
            else:
                return 0.2 + 0.8 * (satisfied_constraints / total_constraints)
        
        predicted_grid = parse_sudoku_answer(solution_str)
        
        if isinstance(ground_truth, str):
            try:
                ground_truth = eval(ground_truth)
            except:
                return 0.0
        
        # Extract original puzzle from extra_info
        original_puzzle = None
        if extra_info and 'puzzle_string' in extra_info:
            original_puzzle = extra_info['puzzle_string']
        
        boxed_content = last_boxed_only_string(solution_str)
        has_boxed = boxed_content is not None
        
        if predicted_grid is None:
            if has_boxed:
                return 0.1
            return 0.0
        
        if predicted_grid == ground_truth:
            return 1.0
        
        return compute_sudoku_score(predicted_grid, original_puzzle)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

