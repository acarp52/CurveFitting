"""
Final project for Stats 251

Authors:    Andrew Carpenter
            David Desrochers
            Cole Rogers
            Connor Coval
            Christopher Arras
"""

def main():
    with open("classB7.dat", "r") as ins:
        data = []
        for line in ins:
            data.append(float(line.rstrip('\n')))

    print(data)

main()