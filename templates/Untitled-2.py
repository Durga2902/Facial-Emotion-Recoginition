def rec(n):
    if n==0:
        return;
    else:
        n*rec(n-1)
        print("*" * n)
n=int(input())
rec(n)