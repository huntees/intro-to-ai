# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def func():

    for x in range (1,6):

        list = [x, x*2, x*3, x*4, x*5]

        for n, i in enumerate(list):
            if i % 2 == 0:
                list[n] = 0

        print(list)

def alt():
    for i in range(1, 6):
        print('\n')
        for j in range(1, 6):
            if i % 2 == 0:
                i = 0
            if j % 2 == 0:
                j = 0
            print(str(i * j).ljust(2), end='   ')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    alt()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
