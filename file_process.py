# f1 = open("dataset/casialist.txt", "r")
# f2 = open("dataset/afterlist.txt", "w")
#
# line = f1.readline()
# line = line.replace("png", "jpg")
# f2.write(line)
#
# while line:
#     line = f1.readline()
#     line = line.replace("png", "jpg")
#     f2.write(line)
#
# f1.close()
# f2.close()

f1 = open("in.txt", "r")
f2 = open("out.txt", "w")

line = f1.readline()
line = line.replace("imageA", "label")
f2.write(line)

while line:
    line = f1.readline()
    line = line.replace("imageA", "label")
    f2.write(line)

f1.close()
f2.close()