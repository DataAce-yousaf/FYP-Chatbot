#write python function to calculate fibonachi
def fibonachi(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonachi(n-1) + fibonachi(n-2)

#use the function
print(fibonachi(10))
 