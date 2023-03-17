def printMatrix(matrix):
    #打印第一行，删除第一行，逆时针转动90度。重复以上步骤，直到矩阵为空。
    result = []
    while matrix:
        result += matrix.pop(0)
        if matrix:
            matrix = [[row[col] for row in matrix] for col in reversed(range(len(matrix[0])))]    # 逆时针旋转
    return result
        
        
        
