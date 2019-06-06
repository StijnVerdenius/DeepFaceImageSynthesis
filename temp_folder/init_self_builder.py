import sys


######### saves all parameters passed to init function als self fields because pycharm doesn't have a function for that


filename = sys.argv[1]

reader = open(filename, "r")

lines = reader.read()

reader.close()

output = ""

for line in lines.split("\n"):
    output += line + "\n"
    if ("def __init__(self" in line):
        elements = line.split("def __init__(self")
        tabs = elements[0]
        fieldnames = elements[1][1:].replace(" ", "").replace(":", "").replace(")", "").split(",")
        for fieldname in fieldnames:
            actual_name = fieldname.split("=")[0]
            output += tabs + "\tself." + actual_name + " = " + actual_name + "\n"

writer = open(filename, "w")

writer.write(output)

writer.close()
