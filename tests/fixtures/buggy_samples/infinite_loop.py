# 边界样例：死循环（run_python 以主脚本方式执行，超时后进程被强杀，REPL 不受影响）。
# 说明：本样例只覆盖"直接子进程被强杀"；exec_py 的 taskkill /T 整树强杀路径
# （为孙进程残留而设）未由本夹具触及。__main__ 守卫使意外 import 本文件不会挂起。
if __name__ == "__main__":
    while True:
        pass
