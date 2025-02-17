import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




def solve_sudoku(grid):
    print("[DEBUG] Starting Sudoku solver...")
    print("Initial grid:")
    print(grid)
    
    def find_empty(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return i, j
        return None
    
    def is_valid(grid, pos, num):
        # Check row
        for j in range(9):
            if grid[pos[0]][j] == num and pos[1] != j:
                return False
        
        # Check column
        for i in range(9):
            if grid[i][pos[1]] == num and pos[0] != i:
                return False
        
        # Check 3x3 box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if grid[i][j] == num and (i, j) != pos:
                    return False
        
        return True
    
    def solve(grid):
        empty = find_empty(grid)
        if not empty:
            return True
        
        row, col = empty
       # print(f"[DEBUG] Trying to fill position ({row}, {col})")
        for num in range(1, 10):
            if is_valid(grid, (row, col), num):
                grid[row][col] = num
                if solve(grid):
                    return True
                grid[row][col] = 0
        
        return False
    
    if solve(grid):
        print("[DEBUG] Sudoku solved successfully!")
        print("Solution:")
        print(grid)
        return grid
    
    print("[ERROR] Could not solve Sudoku")
    return None



