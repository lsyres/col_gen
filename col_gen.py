from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt

# 数据
# width = 218
# size = np.array([81,70,68])
# amount = np.array([44,3,48])

width = 115
size = np.array([25,40,50,55,70])
amount = np.array([50,36,24,8,30])

# 初始化
col_num = len(size)
c = [1]*col_num
b = amount.T
A = np.eye(col_num)
obj_value = []

# 变量
m=Model(name='cutstock')
# x = m.var_list(col_num,name="x")
x = m.continuous_var_list(col_num,name="x")
# x = m.integer_var_list(col_num,name="x")
# 约束
# m.add_constraints(A@x >=b)
cons= m.add_constraints([np.dot(A[i,:],x) >=b[i] for i in range(len(amount))],names='cons')
# 目标
obj = sum(x)
m.set_objective('min',obj)
# 求解主问题
# sol_m = m.solve(log_output = True)
sol_m = m.solve()
obj_value.append(sol_m.get_objective_value())
m.print_solution()

while True:
    # 子问题：Knapsack Problem
    pi =np.array(m.dual_values(cons))  # 子问题的目标函数系数c
    s= Model(name='knapsack')
    # 子问题变量
    y = s.integer_var_list(len(pi),name='y')
    # 子问题约束
    cons_sub = s.add_constraint(np.dot(size,y)<=width)
    # 子问题目标函数
    obj_sub = pi@y
    s.set_objective('max',obj_sub)
    sol_s = s.solve()
    a = np.array(sol_s.get_values(y)) # solution a is a list

    # 更新主问题
    # 新增一列
    col_num +=1  # 增加一个x
    A = np.c_[A,a] # 增加一个列，即增加一个pattern
    # 变量
    m=Model(name='cutstock_update')
    x = m.continuous_var_list(col_num,name="x")
    # 约束
    cons= m.add_constraints([np.dot(A[i,:],x) >=b[i] for i in range(len(amount))],names='cons')
    # 目标
    obj = sum(x)
    m.set_objective('min',obj)
    # 求解主问题
    sol_m = m.solve(log_output = True)
    obj_value.append(sol_m.get_objective_value())
    m.print_solution()
    if obj_value[-1] == obj_value[-2]:
        break

print(obj_value)
plt.plot(obj_value)
plt.show()