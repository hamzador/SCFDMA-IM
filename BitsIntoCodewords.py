def BitsIntoCodewords(InputBinaryS, m, m1, N, M, n, k, g, Frames):
    codwordBit = []
    val = 0
    while val <= (len(InputBinaryS) - 1):
        temp = str(InputBinaryS[val]) + str(InputBinaryS[val + 1]) # variabl to concatinate 2 bits as strings
        codwordBit.append(temp)                                    #Add the 00||01||11||10 to the list of each two string
        val += 2
    codwords = [0] * len(codwordBit)
    for val in range(len(codwordBit)):
        if codwordBit[val] == "00":
            codwords[val] = 1 + 1j
        elif codwordBit[val] == "01":
            codwords[val] = -1 + 1j
        elif codwordBit[val] == "11":
            codwords[val] = -1 - 1j
        else:
            codwords[val] = 1 - 1j

    return codwords