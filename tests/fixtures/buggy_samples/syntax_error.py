# 边界样例：语法错误（run_python 应返回非零退出码，stderr 含 SyntaxError）。
def broken(:
    print("syntax error sample")
