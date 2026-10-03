# 边界样例：语法错误（run_python 应返回退出码 1，stderr 含 SyntaxError）。
# 第 3 行在解析阶段即失败，第 4 行不可达——样例只用于触发编译错误，勿修复。
def broken(:
    print("syntax error sample")
