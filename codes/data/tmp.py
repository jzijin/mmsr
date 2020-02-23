
tmp = {}
with open("./identity_CelebA.txt", 'r') as f:
    for line in f.readlines():
        line = line.rstrip()
        tmp[line.split()[0]] = line.split()[1]
    f.close()
print(tmp)
    # res = f.readlines()[0]
    # print(res)