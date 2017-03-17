import polygons as p
import numpy as np
import matplotlib.pyplot as plt


def normalize(a):
    if np.linalg.norm(a) == 0:
        return a
    return a / np.linalg.norm(a)


def wireVector(w):
    return np.array((w[2] - w[0], w[3] - w[1]))


def wireLength(w):
    return np.linalg.norm(wireVector(w))


def wireHead(w):
    return np.array(w[0:2])


def wireTail(w):
    return np.array(w[2:4])


def dot(a, b):
    return np.sum(a * b)


def tripletCosine(a, b, c):
    v1 = normalize(b - a)
    v2 = normalize(c - b)
    return dot(v1, v2)

# intersects segment/line a1-a2 with segment b2 b2
# checks defines whether the segment is limited (True) or
# unlimited(line,False) on that side


def intersect(a1, a2, b1, b2,
              a1check, a2check, b1check, b2check,
              parallelSensitivity=1E-10):
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)
    mc1 = (a2 - a1).reshape((2, 1))
    mc2 = (b1 - b2).reshape((2, 1))
    m = np.hstack((mc1, mc2))
    d = (b1 - a1).reshape((2, 1))
    if np.abs(np.linalg.det(m)) < parallelSensitivity:
        return False
    coeffs = np.linalg.inv(m).dot(d)

    if a1check and coeffs[0] < 0 or a2check and coeffs[0] > 1 or \
       b1check and coeffs[1] < 0 or b2check and coeffs[1] > 1:
        return False
    return coeffs[0] * a2 + (1 - coeffs[0]) * a1


def intersectSegments(a1, a2, b1, b2, parallelSensitivity=1E-10):
    return intersect(a1, a2, b1, b2,
                     True, True, True, True,
                     parallelSensitivity)


def intersectLines(a1, a2, b1, b2, parallelSensitivity=1E-10):
    return intersect(a1, a2, b1, b2,
                     False, False, False, False,
                     parallelSensitivity)


def getParallel(w, center, offset):
    global inv, ninv
    p1 = np.array(w[0:2])
    p2 = np.array(w[2:4])
    v = p1 - p2
    # c = p1 - np.array(center)
    vo = offset * normalize(np.array((v[1], -v[0])))
    # if sum(vo * c) < 0:
    #    vo = -vo
    #    inv = inv + 1
    # else:
    #    ninv = ninv + 1
    return (p1 + vo, p2 + vo)

# http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order


def isPolygonClockwise(po):
    s = 0
    for w in po:
        s += (w[2] - w[0]) * (w[3] + w[1])
    return s > 0


def plotWire(w):
    plt.plot(w[0::2], w[1::2])


def plotWireColor(w, color):
    plt.plot(w[0::2], w[1::2], color=color)


def plotLine(l):
    plt.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]))


def drawPolygons(pols):
    for pol in pols:
        for w in pol:
            plotWire(w)
    plt.show()


def drawSingleColor(pols, color):
    for pol in pols:
        for w in pol:
            plt.plot(w[0::2], w[1::2], color=color)


def vertsToWires(verts):
    wires = []
    for iv in range(0, len(verts) - 1):
        wire = (verts[iv][0], verts[iv][1],
                verts[iv + 1][0], verts[iv + 1][1])
        wires.append(wire)
    wires = [(verts[-1][0], verts[-1][1],
              verts[0][0], verts[0][1]), ] + wires
    return tuple(wires)


def removeLoops(averts, minLength):
    for i in range(0, len(averts)):
        for j in range(i + 1, len(averts)):
            if np.linalg.norm(averts[i] - averts[j]) < 1:
                del averts[i + 1:j + 1]
                print("removed loop")
                return True
    return False


def removeIntersections(averts):
    for i in range(0, len(averts)):
        for j in range(i + 3, len(averts)):
            if intersectSegments(averts[i], averts[i + 1], averts[j - 1], averts[j]) is not False:
                print(averts[i + 1], averts[i + 1], averts[j - 1], averts[j])
                averts[i + 1] = averts[j]
                del averts[i + 2:j + 1]
                print("removed intersection")
                return True
    return False


def simplify(pols, minAngle, minLength):
    newPols = []
    for polygon in pols:
        verts = []
        verts.append(np.array(polygon[0][0:2]))
        for w in polygon:
            v = np.array(w[0:2])
            if np.linalg.norm(v - verts[-1]) > minLength:
                verts.append(v)
        if len(verts) < 3:
            continue

        averts = [verts[0], ]
        for iv in range(1, len(verts) - 1):
            if tripletCosine(verts[iv - 1], verts[iv], verts[iv + 1]) < np.cos(minAngle):
                averts.append(verts[iv])
        while(removeLoops(averts, minLength)):
            pass
        while(removeIntersections(averts)):
            pass
        if len(averts) < 3:
            continue

        wires = vertsToWires(averts)
        newPols.append(wires)
    return newPols


def getOffseted(pols, offset=100):
    print("{} polygons".format(len(p.data)))
    newPols = []
    for polygon in pols:
        wiree = polygon + (polygon[0],)
        center = np.mean(np.array(wiree), axis=0)[0:2]
        verts = []
        for i in range(0, len(wiree) - 1):
            if wiree[i][2:4] != wiree[i + 1][0:2]:
                raise Exception("polygon verteces are not consecutive")

            a = getParallel(wiree[i], center, offset)
            b = getParallel(wiree[i + 1], center, offset)
            inter = intersectLines(a[0], a[1], b[0], b[1])
            if inter is False:
                plotWire(wiree[i])
                plotWire(wiree[i + 1])
                plotLine(a)
                plotLine(b)
                plt.show()
                msg = "original\n {} \n{}\n parallel lines\n {}\n{}"
                raise Exception(msg.format(wiree[i], wiree[i + 1], a, b))
                inter = a[1]

            verts.append(inter)
        wires = vertsToWires(verts)

        goodPoly = not isPolygonClockwise(wires)

        for i in range(0, len(wires)):
            if dot(wireVector(wires[i]), wireVector(polygon[i])) < 0:
                print("polygon direction inversion")
                goodPoly = False
                break

        if goodPoly:
            newPols.append(tuple(wires))

    return newPols


def compare():
    symp = simplify(p.data, 0.5 * np.pi / 180, 10000)
    off = getOffseted(symp, 1000)
    drawSingleColor(p.data, "black")
    drawSingleColor(symp, "blue")
    drawSingleColor(off, "red")
    plt.show()
