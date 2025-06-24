def euclidean_distance(x, y):
    #Nếu kích thước vector không giống nhau in ra None
    if len(x) != len(y):
        return None
    #Tính khoảng cách euclidean (bản chất là trừ vector)
    squared_distance = 0
    for i in range(len(x)):
        squared_distance += (x[i] - y[i]) ** 2
    return squared_distance ** 0.5