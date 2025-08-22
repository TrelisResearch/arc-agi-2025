# Task: b7249182
# Model: Mistral-Large-Instruct-2407
# Length: 4,803 characters
# Train correct: 3/3
# Test correct: 1/1

def transform(grid):
    out = [row[:] for row in grid]
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    seeds = []
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] != 0:
                seeds.append((r, c, grid[r][c]))
    if len(seeds) != 2:
        return out
    if seeds[0][1] == seeds[1][1]:
        seeds.sort(key=lambda x: x[0])
        r_top, col, color_top = seeds[0]
        r_bot, _, color_bot = seeds[1]
        mid = (r_top + r_bot) // 2
        top_primary_end = mid - 1
        bottom_primary_start = mid + 2
        for r in range(r_top, min(top_primary_end + 1, nrows)):
            if 0 <= r < nrows and 0 <= col < ncols:
                out[r][col] = color_top
        r_bar = top_primary_end
        for j in range(col - 2, col + 3):
            if 0 <= r_bar < nrows and 0 <= j < ncols:
                out[r_bar][j] = color_top
        r_ep = top_primary_end + 1
        for j in [col - 2, col + 2]:
            if 0 <= r_ep < nrows and 0 <= j < ncols:
                out[r_ep][j] = color_top
        for r in range(r_bot, bottom_primary_start - 1, -1):
            if 0 <= r < nrows and 0 <= col < ncols:
                out[r][col] = color_bot
        r_bar = bottom_primary_start
        for j in range(col - 2, col + 3):
            if 0 <= r_bar < nrows and 0 <= j < ncols:
                out[r_bar][j] = color_bot
        r_ep = bottom_primary_start - 1
        for j in [col - 2, col + 2]:
            if 0 <= r_ep < nrows and 0 <= j < ncols:
                out[r_ep][j] = color_bot
    elif seeds[0][0] == seeds[1][0]:
        seeds.sort(key=lambda x: x[1])
        row, c_left, color_left = seeds[0]
        _, c_right, color_right = seeds[1]
        mid = (c_left + c_right) // 2
        left_primary_end = mid - 1
        right_primary_start = mid + 2
        for j in range(c_left, min(left_primary_end + 1, ncols)):
            if 0 <= row < nrows and 0 <= j < ncols:
                out[row][j] = color_left
        for r in [row - 1, row + 1]:
            if 0 <= r < nrows and 0 <= left_primary_end < ncols:
                out[r][left_primary_end] = color_left
        for r in [row - 2, row + 2]:
            for j in [left_primary_end, left_primary_end + 1]:
                if 0 <= r < nrows and 0 <= j < ncols:
                    out[r][j] = color_left
        for j in range(right_primary_start, c_right + 1):
            if 0 <= row < nrows and 0 <= j < ncols:
                out[row][j] = color_right
        for r in [row - 1, row + 1]:
            if 0 <= r < nrows and 0 <= right_primary_start < ncols:
                out[r][right_primary_start] = color_right
        for r in [row - 2, row + 2]:
            for j in [right_primary_start - 1, right_primary_start]:
                if 0 <= r < nrows and 0 <= j < ncols:
                    out[r][j] = color_right
    else:
        seeds.sort(key=lambda x: (x[0], x[1]))
        r_top, c_left, color_top_left = seeds[0]
        r_bot, c_right, color_bot_right = seeds[1]
        mid_r = (r_top + r_bot) // 2
        mid_c = (c_left + c_right) // 2
        top_primary_end = mid_r - 1
        bottom_primary_start = mid_r + 2
        left_primary_end = mid_c - 1
        right_primary_start = mid_c + 2
        for r in range(r_top, min(top_primary_end + 1, nrows)):
            if 0 <= r < nrows and 0 <= c_left < ncols:
                out[r][c_left] = color_top_left
        for j in range(c_left, min(left_primary_end + 1, ncols)):
            if 0 <= r_top < nrows and 0 <= j < ncols:
                out[r_top][j] = color_top_left
        r_bar = top_primary_end
        for j in range(left_primary_end, right_primary_start + 1):
            if 0 <= r_bar < nrows and 0 <= j < ncols:
                out[r_bar][j] = color_top_left
        for r in range(top_primary_end + 1, bottom_primary_start):
            for j in [left_primary_end, right_primary_start]:
                if 0 <= r < nrows and 0 <= j < ncols:
                    out[r][j] = color_top_left
        for r in range(bottom_primary_start, r_bot + 1):
            if 0 <= r < nrows and 0 <= c_right < ncols:
                out[r][c_right] = color_bot_right
        for j in range(right_primary_start, c_right + 1):
            if 0 <= r_bot < nrows and 0 <= j < ncols:
                out[r_bot][j] = color_bot_right
        r_bar = bottom_primary_start
        for j in range(left_primary_end, right_primary_start + 1):
            if 0 <= r_bar < nrows and 0 <= j < ncols:
                out[r_bar][j] = color_bot_right
        for r in range(bottom_primary_start - 1, top_primary_end, -1):
            for j in [left_primary_end, right_primary_start]:
                if 0 <= r < nrows and 0 <= j < ncols:
                    out[r][j] = color_bot_right
    return out