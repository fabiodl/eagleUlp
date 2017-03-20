import numpy as np
import matplotlib.pyplot as plt
from svg.path import parse_path

#'fiat411r_tractor_3_simplify_plain.svg'


def normalize(a):
    if np.linalg.norm(a) == 0:
        return a
    return a / np.linalg.norm(a)


def dot(a, b):
    return np.sum(a * b)


def complexToPt(x):
    return np.array((np.real(x), -np.imag(x)))


def tripletCosine(a, b, c):
    a = complexToPt(a)
    b = complexToPt(b)
    c = complexToPt(c)
    v1 = normalize(b - a)
    v2 = normalize(c - b)
    return dot(v1, v2)


def removeParallel(verts, minAngle):
    averts = [verts[0], ]
    for iv in range(1, len(verts) - 1):
        if tripletCosine(averts[-1], verts[iv], verts[iv + 1]) < np.cos(minAngle):
            averts.append(verts[iv])
    averts.append(verts[-1])
    return averts


def readFile(filename="plain.svg", minAngle=3 * np.pi / 180):
    import xml.etree.ElementTree
    e = xml.etree.ElementTree.parse(filename).getroot()
    allPts = []
    allWidths = []
    tl = 0
    for c in e.find("{http://www.w3.org/2000/svg}g"):
        if c.tag != "{http://www.w3.org/2000/svg}path":
            continue
        pathStr = c.get("d")
        path = parse_path(pathStr)
        pts = [path.point(x) for x in np.linspace(0, 1, 100)]
        pts = removeParallel(pts, minAngle)
        allPts.append(pts)
        tl = tl + len(pts)

        style = c.get("style")
        for token in style.split(";"):
            prop = token.split(":")
            if prop[0] == "stroke-width":
                allWidths.append(float(prop[1]))

    print("{} wires, for a total of {} pts".format(len(allPts), tl))
    if len(allPts) != len(allWidths):
        print("WARNING: NOT ALL THE PATHS HAVE A WIDTH")
    return allPts, allWidths

    # path = parse_path(totalData)
    # pts = [path.point(x) for x in np.linspace(0, 1, 10000)]
    # plt.plot(np.real(pts), -np.imag(pts))


def plotPaths(allPts):
    for pts in allPts:
        plt.plot(np.real(pts), -np.imag(pts))
    plt.show()


def ptsToScr(outfilename, allPts, allWidths, layer="tPlace"):
    f = open(outfilename, "w")
    headCommands = (
        "set wire_bend 2",
        "change layer {}".format(layer),
        "grid mm",
    )
    re = np.real

    def im(x):
        return -np.imag(x)

    for c in headCommands:
        f.write(c)
        f.write(";\n")

    for p, w in zip(allPts, allWidths):

        f.write("wire {} ".format(w))
        for i in range(0, len(p)):
            msg = "({} {}) "
            f.write(msg.format(re(p[i]), im(p[i])))
        f.write(";\n")


def scaleByHeight(pts,  width, desired):
    ptsFlat = sum(pts, [])
    scale = desired / (np.max(np.imag(ptsFlat)) - np.min(np.imag(ptsFlat)))
    print("scale ", scale)

    return [np.multiply(p, scale) for p in pts], np.multiply(width, scale)

# ap=readFile()
# sp=scaleByHeight(ap,0.3*25.4)


# plotPaths(readFile())
# ptsToScr("tractor.scr", sp, 0.127)

def svgToScr(infilename="plain.svg",
             outfilename="tractor.scr",
             desiredHeight=8,
             layer="tPlace"
             ):
    print("reading ", infilename)
    ap, aw = readFile(infilename)
    sp, sw = scaleByHeight(ap, aw, desiredHeight)
    ptsToScr(outfilename, sp, sw, layer)


svgToScr()
